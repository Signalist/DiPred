"""Full-grid, complete20 empirical probability scores; no independent-pixel CI."""
import torch


def empirical_crps(paths,truth):
    n=len(paths)
    ordered=paths.sort(0).values
    weight=(2*torch.arange(n,device=paths.device,dtype=paths.dtype)-n+1)
    shape=[n]+[1]*(paths.ndim-1)
    return (paths-truth).abs().mean(0)-(ordered*weight.reshape(shape)).sum(0)/(n*n)


def energy_score(paths,truth):
    """RMS Euclidean norm across all C*T*H*W; all ordered member pairs."""
    p=paths.reshape(len(paths),-1).double()
    y=truth.reshape(-1).double()
    first=(p-y).square().mean(1).sqrt().mean()
    norm=p.square().sum(1)
    distances=((norm[:,None]+norm[None,:]-2*p@p.T).clamp_min(0)/p.shape[1]).sqrt()
    distances.fill_diagonal_(0.)
    return first-.5*distances.mean()


def acf(paths,lag):
    # [member,channel,time,y,x], pool member/time/space; channel means removed.
    a=paths[:,:,:-lag].movedim(1,0).reshape(2,-1).double()
    b=paths[:,:,lag:].movedim(1,0).reshape(2,-1).double()
    a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
    return (a*b).sum(1)/(a.square().sum(1)*b.square().sum(1)).sqrt().clamp_min(1e-30)


def consecutive_event(field,threshold):
    # Input N,C,T,H,W; only v-channel physical spatial RMS.
    active=field[:,1].square().mean((-1,-2)).sqrt()>threshold
    return active.unfold(-1,3,1).all(-1).any(-1)


@torch.no_grad()
def scores(paths,truth,std,event_threshold):
    paths,truth=paths.double(),truth.double()
    mean=paths.mean(0)
    delta=mean-truth
    normalized=paths/std;target=truth/std
    crps=empirical_crps(paths,truth)
    ncrps=empirical_crps(normalized,target)
    forecast=torch.stack([acf(paths,k) for k in range(1,20)])
    reference=torch.stack([acf(truth[None],k) for k in range(1,20)])
    probability=consecutive_event(paths,event_threshold).double().mean()
    observed=consecutive_event(truth[None],event_threshold)[0].double()
    output=dict(mse=float(delta.square().mean()),mae=float(delta.abs().mean()),
        crps=float(crps.mean()),crps_standardized=float(ncrps.mean()),
        energy=float(energy_score(paths,truth)),energy_standardized=float(energy_score(normalized,target)),
        temporal_acf_rmse=float((forecast-reference).square().mean().sqrt()),
        event_probability=float(probability),event_observed=float(observed),
        event_brier=float((probability-observed).square()),
        mse_u=float(delta[0].square().mean()),mse_v=float(delta[1].square().mean()),
        crps_u=float(crps[0].mean()),crps_v=float(crps[1].mean()))
    return output,forecast.cpu().numpy(),reference.cpu().numpy()
