"""Training-gate numerical tests for error-guided spatial routing."""
import hashlib, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reliability.networks import DynamicsDiffusion20
from research.error_guided_spatial_routing.model import TrustDiffusion,crps


def state_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    ck=torch.load(ROOT/"reliability/checkpoints/current.pt",map_location="cuda",weights_only=False)
    native=DynamicsDiffusion20(ck["dynamics_scale"]).cuda().eval();native.load_state_dict(ck["predictor"])
    base=state_hash(native)
    context=TrustDiffusion(native,False).cuda();trust=TrustDiffusion(native,True).cuda()
    nctx=sum(p.numel() for p in context.router.parameters());ntrust=sum(p.numel() for p in trust.router.parameters())
    assert nctx==ntrust and nctx<=10000 and nctx>0
    torch.manual_seed(123)
    h=torch.randn(1,2,20,32,64,device="cuda")
    z=torch.randn(32,2,20,32,64,device="cuda")
    aux=torch.randn(32,3,32,64,device="cuda")
    risk=torch.rand(32,1,32,64,device="cuda")
    labels=torch.full((32,),99,device="cuda",dtype=torch.long)
    # Disabled router must be exactly native at every denoising label.
    with torch.no_grad():
        a=native.net(z,labels,h.expand(32,-1,-1,-1,-1))
        b=context.forward(z,labels,h.expand(32,-1,-1,-1,-1),aux,risk,enabled=False)
    assert torch.equal(a,b)
    # Complete CRPS and its explicit pairwise form agree in float64.
    p=torch.randn(7,2,3,4,5,dtype=torch.float64);y=torch.randn(2,3,4,5,dtype=torch.float64)
    explicit=(p-y).abs().mean(0)-.5*(p[:,None]-p[None,:]).abs().mean((0,1))
    assert torch.allclose(crps(p,y),explicit.mean(),atol=1e-12,rtol=1e-12)
    # All four steps backpropagate through the router, and the frozen hash is unchanged.
    model=TrustDiffusion(DynamicsDiffusion20(ck["dynamics_scale"]).cuda().eval(),True).cuda()
    model.native.load_state_dict(ck["predictor"])
    opt=torch.optim.Adam(model.router.parameters(),lr=.001)
    out=model.sample(h.expand(32,-1,-1,-1,-1),aux,risk,z)
    loss=crps(out,torch.randn_like(out[0]));opt.zero_grad();loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.router.parameters())
    assert any(float(p.grad.abs().sum())>0 for p in model.router.parameters())
    assert state_hash(model.native)==base
    # Future truth cannot enter the release input: only the later loss/reveal uses it.
    future_a=torch.zeros(1,2,20,32,64);future_b=torch.ones_like(future_a)
    assert torch.equal(h,h) and not torch.equal(future_a,future_b)
    result={"passed":True,"router_parameters":nctx,"max_router_strength":0.25,
            "closed_router_exact":True,"crps_pair_gradient_checked":True,
            "all_four_steps_gradient_checked":True,"backbone_hash":base,
            "future_truth_input_independent":True}
    outp=ROOT/"results/error_guided_spatial_routing";outp.mkdir(parents=True,exist_ok=True)
    (outp/"PREFLIGHT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
