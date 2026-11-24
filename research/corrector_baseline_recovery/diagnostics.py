"""Fixed synthetic and real-window diagnostics in the actual release units."""
from types import SimpleNamespace

import torch

from research.corrector.run import dump, table
from research.corrector_predictability.data import EXPANDED, Split, normalization, sources
from reliability.backbone import configure

from .legacy import OUT
from .model import BlockDelta, input_fields, issue, network
from .crossfit_corrector import BlockResidualCorrector


def update(model, optimizer, batch, labels):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(*batch)-labels).square().mean()
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.detach(), gradient.detach()


def run(data, mean, units, scale, out):
    p, y = data.p[:4].double(), data.y[:4].double()
    physical_truth = y-p
    assert torch.equal(issue(p,physical_truth,0.),p)
    torch.testing.assert_close(issue(p,physical_truth,1.),y,atol=1e-15,rtol=0.)
    prediction = .35 * physical_truth + .001
    mse_error = float(((issue(p,prediction,1.)-y).square().mean() -
                       (prediction-physical_truth).square().mean()).abs())
    assert mse_error < 1e-16
    x, pn = input_fields(data,mean,units)
    for normalized, physical in ((x,data.x),(pn,data.p)):
        torch.testing.assert_close(normalized*units+mean,physical,atol=2e-7,rtol=0.)
    original = BlockResidualCorrector().cuda()
    torch.nn.init.normal_(original.residual[-1].weight,std=.015)
    adapted = BlockDelta().cuda()
    adapted.load_state_dict(original.state_dict())
    with torch.no_grad():
        full = original(x[:4],pn[:4])
        as_delta = adapted(x[:4],pn[:4])
        torch.testing.assert_close(pn[:4]+as_delta,full,rtol=0.,atol=2e-7)
    results, curves = [], []
    stages = (0,50,100,250,500,750,1000)
    for label in ("bias","channel_scale","real_1","real_4"):
        count = 1 if label=="real_1" else 4
        subset = SimpleNamespace(x=data.x[:count],p=data.p[:count])
        fields = input_fields(subset,mean,units)
        truth = data.y[:count]-data.p[:count]
        if label == "bias":
            truth = torch.tensor([.008,-.004],device=p.device).view(1,2,1,1,1).expand_as(truth)
        elif label == "channel_scale":
            truth = (subset.p-mean)*torch.tensor([.08,-.16],device=p.device).view(1,2,1,1,1)
        target = truth/scale
        for kind in ("Point","Block"):
            torch.manual_seed(42)
            head = network(kind,p.device)
            opt = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=999,eta_min=1e-6)
            values = []
            for step in range(1001):
                if step in stages:
                    with torch.no_grad():
                        mse = float((head(*fields)-target).square().mean())*scale**2
                    values.append(mse)
                    curves.append(dict(probe=label,windows=count,architecture=kind,step=step,physical_mse=mse))
                if step<1000:
                    loss, grad = update(head,opt,fields,target)
                    sched.step()
            reduction = 100*(1-values[-1]/values[0])
            assert torch.isfinite(torch.stack((loss,grad))).all() and reduction>0
            row = dict(probe=label,architecture=kind,windows=count,initial_mse=values[0],
                       final_mse=values[-1],reduction_pct=reduction)
            results.append(row)
            print("DIAGNOSTIC",row,flush=True)
    table(out/"diagnostic_curves.csv",curves)
    passed = all(r["reduction_pct"] >= 90. for r in results if r["probe"] in ("bias","channel_scale"))
    audit = dict(passed=passed,zero_exact=True,truth_recovers_y=True,
                 normalized_physical_roundtrip=True,old_block_returns_q=True,
                 residual_vs_forecast_mse_error=mse_error,
                 examples=[{k:c[k] for k in ("file","start","history","target")} for c in data.cases[:4]],
                 results=results)
    dump(out/"diagnostic_audit.json",audit)
    assert passed, "Synthetic predictable residual diagnostic did not pass"
    return audit


if __name__ == "__main__":
    configure()
    manifest, _ = sources()
    data = Split(manifest,"train",EXPANDED)
    mean, units, scale, _ = normalization(data.x.device)
    run(data,mean,units,scale,OUT)
