"""Same four-call graph and dynamics inputs, trained using terminal MSE only."""
import os,sys,time,math,argparse
from pathlib import Path
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import torch
from structure import W,OUT,setup,load,save,dump,norm,model,capture,digest,SHARED,MANIFEST
sys.path.insert(0,str(W))
from physical20_augmented_dynamics import AugmentedDynamics20
from common import records,DATA

def forward(m,h):
    x=torch.zeros_like(h)
    for k in reversed(range(4)):
        a=m.abar[k];t=torch.full((len(h),),k*33,device=h.device,dtype=torch.long)
        clean=m.net(x,t,h);eps=(x-a.sqrt()*clean)/(1-a).sqrt()
        prior=m.abar[k-1] if k else h.new_tensor(1.)
        x=prior.sqrt()*clean+(1-prior).sqrt()*eps
    return x

def train():
    import h5py,numpy as np,hashlib
    dest=OUT/'deterministic4';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'model6000.pt').exists(),'Completed endpoint already exists'
    fields=[]
    for r in records('train'):
        with h5py.File(DATA/r['file'],'r') as f:a=np.stack([f[c][:200,::2,::2] for c in ('u','v')]).astype('float32')
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['sha256_first200']
        fields.append(torch.from_numpy(a))
    fields=torch.stack(fields).cuda();fields=(fields-norm()['mean'])/norm()['std']
    h=torch.cat([fields[:,:,s:s+20] for s in range(0,161,5)])
    y=torch.cat([fields[:,:,s+20:s+40] for s in range(0,161,5)])
    del fields
    scale=load(SHARED)['dynamics_scale'].cuda();torch.manual_seed(42)
    m=AugmentedDynamics20(scale).cuda();ref=model()
    assert set(m.state_dict())==set(ref.state_dict())
    assert all(v.shape==ref.state_dict()[k].shape for k,v in m.state_dict().items())
    # Independently verify the deployed graph equals our differentiable graph.
    a=forward(ref,h[:2]);b=ref.rollout(h[:2],500000,sign=0)
    difference=float((a-b).abs().max());assert difference==0
    del ref
    dump(dest/'protocol.json',dict(seed=42,updates=6000,checkpoints=[1500,6000],batch=8,fit_windows=len(h),fit_files=[r['file'] for r in records('train')],frames=[0,200],loss='terminal future standardized MSE through four shared calls; no injected noise',inference_graph_max_error=difference,parameters=sum(p.numel() for p in m.parameters()),limitations='1500 updates approximately matches training network-call count;6000 matches optimizer-update count but spends more training compute than native one-call denoising loss. Data schedule uses a dedicated seed42 generator.',manifest_sha256=digest(MANIFEST),shared_sha256=digest(SHARED)))
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5);g=torch.Generator(device='cuda').manual_seed(42)
    start=time.perf_counter();logs=[]
    for step in range(1,6001):
        ix=torch.randint(len(h),(8,),device='cuda',generator=g)
        lr=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/6000))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad();loss=(forward(m,h[ix])-y[ix]).square().mean();assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(grad);opt.step()
        if step==1 or step%250==0:
            row=dict(step=step,loss=float(loss),seconds=time.perf_counter()-start);logs.append(row);dump(dest/'training.json',logs);print(row,flush=True)
        if step in (1500,6000):save(dict(model=m.state_dict(),step=step,seed=42),dest/f'model{step}.pt')
    dump(dest/'complete.json',dict(endpoint_sha256=digest(dest/'model6000.pt'),seconds=time.perf_counter()-start))

@torch.no_grad()
def prepare(split):
    d=load(OUT/f'{split}.pt');m=model();new={}
    for steps in (1500,6000):
        m.load_state_dict(load(OUT/'deterministic4'/f'model{steps}.pt','cuda')['model']);features=[];pred=[]
        for i in range(0,len(d['x']),8):
            p,f=capture(m,d['x'][i:i+8].cuda());features.append(f['F_last'].cpu());pred.append(p.cpu())
        new[f'F_det4_{steps}']=torch.cat(features)
        save(dict(p=torch.cat(pred),meta=d['meta']),OUT/f'{split}_det4_{steps}.pt')
    d['features'].update(new);save(d,OUT/f'{split}.pt')
    print('DET4 FEATURES',split,flush=True)

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['train','prepare']);a.add_argument('--split',default='train');args=a.parse_args()
    train() if args.action=='train' else prepare(args.split)
