"""Physical cross-window associations and intervention-based receptive fields."""
import torch

from research.corrector_residual_optimization.fit import track_weights
from research.corrector_simple_baseline.fit import table

from .model import ARMS,SLOTS,encode,predict


def pearson(a,b,aa,bb,ab):
    return (ab-a*b)/((aa-a*a).clamp_min(0)*(bb-b*b).clamp_min(0)).sqrt().clamp_min(1e-30)


@torch.inference_mode()
def correlations(data,features,out):
    target = data.y.double()-data.p.double()
    w = track_weights(data)
    spatial, temporal, whole = [],[],[]
    all_regimes = sorted({c["re"] for c in data.cases})
    for slot,field in zip(SLOTS,features):
        f = field.double()
        def stats(a,b,wa):
            def reduce(t):
                return float((wa*t.reshape(len(wa),-1).mean(1)).sum())
            m,n,x,y,c = (reduce(t) for t in (a,b,a.square(),b.square(),a*b))
            corr = (c-m*n)/max(((x-m*m)*(y-n*n))**.5,1e-30)
            return dict(pearson=corr,cosine=c/max((x*y)**.5,1e-30),inner=c,
                        previous_mean=m,current_mean=n,previous_rms=x**.5,current_rms=y**.5)
        from .run import subset
        for regime in ("all",*all_regimes):
            ix = [i for i,c in enumerate(data.cases) if regime=="all" or c["re"]==regime]
            a,b = f[ix],target[ix]
            wa = track_weights(subset(data,ix))
            whole.append(dict(source=slot,re=regime,channel="all",**stats(a,b,wa)))
            for c,label in enumerate(("u","v")):
                whole.append(dict(source=slot,re=regime,channel=label,**stats(a[:,c],b[:,c],wa)))
        weight = w.view(-1,1,1,1,1)
        out_rows = {}
        for key,value in (("a",f),("b",target),("aa",f.square()),("bb",target.square()),("ab",f*target)):
            out_rows[key] = (value*weight).sum(0).mean(1)
        a,b,aa,bb,ab = (out_rows[k] for k in ("a","b","aa","bb","ab"))
        p = pearson(a,b,aa,bb,ab).cpu().numpy()
        cv = (ab/(aa*bb).sqrt().clamp_min(1e-30)).cpu().numpy()
        inner = ab.cpu().numpy()
        spatial.extend(dict(source=slot,channel=channel,y=y,x=x,pearson=p[c,y,x],cosine=cv[c,y,x],
                            inner=inner[c,y,x]) for c,channel in enumerate(("u","v"))
                       for y in range(32) for x in range(64))
        moments = [torch.zeros(20,device=f.device,dtype=torch.float64) for _ in range(4)]
        product = torch.zeros((20,20),device=f.device,dtype=torch.float64)
        for start in range(0,len(w),16):
            weights = w[start:start+16].view(-1,1,1,1,1)
            a,b = f[start:start+16],target[start:start+16]
            for accumulator,value in zip(moments,(a,b,a.square(),b.square())):
                accumulator += (value*weights).sum(0).mean((0,2,3))
            a = (a*weights.sqrt()).permute(2,0,1,3,4).reshape(20,-1)
            b = (b*weights.sqrt()).permute(2,0,1,3,4).reshape(20,-1)
            product += (a@b.T)/(2*32*64)
        a,b,aa,bb = moments
        matrix = pearson(a[:,None],b[None,:],aa[:,None],bb[None,:],product).cpu().numpy()
        covector = (product/(aa[:,None]*bb[None,:]).sqrt().clamp_min(1e-30)).cpu().numpy()
        temporal.extend(dict(source=slot,previous_lead=i+1,current_lead=j+1,pearson=matrix[i,j],
                             cosine=covector[i,j],inner=float(product[i,j])) for i in range(20) for j in range(20))
    for file,values in (("residual_correlations",whole),("spatial_correlations",spatial),
                        ("temporal_correlations",temporal)):
        table(out/f"{file}.csv",values)


@torch.inference_mode()
def dependency(dev,base,feedback,cfg,selected,out):
    from .run import checked_model
    plot = cfg["case"]
    where = [i for i,c in enumerate(dev.cases) if (c["file"],c["start"])==(plot["file"],plot["start"])]
    assert len(where)==1
    x = base[where]
    physical = [v[where] for v in feedback]
    scale = cfg["rms"]
    interventions = [(name,[int(i==j) for i in range(3)]) for j,name in enumerate(SLOTS)]
    rows = []
    for name,mask in ARMS.items():
        record = selected["models"][name]["selected"]
        net = checked_model(12,record,out,x.device)
        memo = predict(net,encode(x,physical,mask,scale),scale)
        available = interventions[:]
        if sum(mask)>=2:
            available += [("joint_true",[1,0,1]),("joint_predicted",[0,1,-1])]
        for kind,vector in available:
            for i in range(20):
                fields = [v.clone() for v in physical]
                for component,sign in zip(fields,vector):
                    component[:,:,i] += scale*.1*sign
                if kind.startswith("joint"):
                    torch.testing.assert_close(fields[0].double()-fields[1].double(),fields[2].double(),
                                               atol=7e-9,rtol=1e-6)
                got = predict(net,encode(x,fields,mask,scale),scale)
                val = (got.double()-memo.double()).square().mean((0,1,3,4)).sqrt().tolist()
                for j,v in enumerate(val):
                    if abs(i-j)>3 or not any(a*b for a,b in zip(mask,vector)):
                        assert v==0.
                    rows.append(dict(arm=name,intervention=kind,previous_lead=i+1,current_lead=j+1,
                                     response_rms=v))
    table(out/"input_dependency.csv",rows)
