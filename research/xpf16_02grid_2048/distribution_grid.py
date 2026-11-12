"""All 0.2%-spaced empirical quantiles and exact sample-CRPS components."""
import torch


def quantiles_and_pair(paths,point,levels,chunk=4096):
    """Sort fixed sample paths per pixel; return residual quantiles and pair sum.

    Pixel chunks avoid the large strided-axis CUDA topk limitation. The pair
    sum is computed before observing the forecast target, so actual CRPS can
    be evaluated later without repeating the sort or leaking future truth.
    """
    count=len(paths)
    flat=paths.flatten(1)
    p=point.flatten()
    result=torch.empty((len(levels),p.numel()),device=paths.device,dtype=paths.dtype)
    pair=torch.empty_like(p,dtype=torch.float64)
    positions=levels.to(paths.device,dtype=torch.float64)*(count-1)
    low=positions.floor().long();high=(low+1).clamp(max=count-1)
    fraction=(positions-low).to(paths.dtype)
    weights=(2*torch.arange(count,device=paths.device,dtype=torch.float64)-count+1)[None]
    for i in range(0,p.numel(),chunk):
        j=min(i+chunk,p.numel())
        ordered=flat[:,i:j].transpose(0,1).contiguous().sort(dim=1).values
        q=ordered[:,low]+(ordered[:,high]-ordered[:,low])*fraction[None]
        result[:,i:j]=(q-p[i:j,None]).transpose(0,1)
        pair[i:j]=((ordered.double()-p[i:j,None].double())*weights).sum(1)
    return result.reshape(len(levels),*point.shape),pair.reshape_as(point)


def sample_crps(paths,point,residual,scale,half_pair,chunk=4096):
    """CRPS of raw/positive-scale samples and their fair estimators."""
    count=len(paths);flat=paths.flatten(1)
    p=point.flatten();y=residual.flatten();g=scale.flatten();pair=half_pair.flatten()
    out=[torch.empty_like(p) for _ in range(4)]
    for i in range(0,p.numel(),chunk):
        j=min(i+chunk,p.numel())
        x=flat[:,i:j].double()-p[i:j].double()[None]
        a=(x-y[i:j].double()[None]).abs().mean(0)
        b=(x*g[i:j].double()[None]-y[i:j].double()[None]).abs().mean(0)
        q=pair[i:j];factor=g[i:j].double()
        scores=(a-q/(count*count),a-q/(count*(count-1)),
                b-factor*q/(count*count),b-factor*q/(count*(count-1)))
        for target,value in zip(out,scores):target[i:j]=value.float()
    return {k:v.reshape_as(point) for k,v in zip(('raw_empirical','raw_fair','proxy_empirical','proxy_fair'),out)}


def rearrange_quantiles(q,chunk=4096):
    """Sort per pixel and report original adjacent crossing/any-crossing."""
    shape=q.shape
    flat=q.flatten(1)
    result=torch.empty_like(flat)
    pair_count=torch.empty(flat.shape[1],device=q.device,dtype=q.dtype)
    any_count=torch.empty_like(pair_count)
    for i in range(0,flat.shape[1],chunk):
        j=min(i+chunk,flat.shape[1])
        block=flat[:,i:j].transpose(0,1).contiguous()
        crossed=block[:,1:]<block[:,:-1]
        pair_count[i:j]=crossed.float().mean(1)
        any_count[i:j]=crossed.any(1).float()
        result[:,i:j]=block.sort(dim=1).values.transpose(0,1)
    return result.reshape(shape),pair_count.reshape(shape[1:]),any_count.reshape(shape[1:])


def quantile_distribution_scores(q,residual):
    """Approximate quantile CRPS and central-interval WIS; q is 499 levels.

    Tau spacing=.002; Q(0) and Q(1) are clamped to endpoint values. Q(.5)
    is the separately trained median for XPF and sample median for noise.
    WIS uses 249 central intervals and is meaningful after rearrangement.
    """
    if len(q)!=499:
        raise ValueError('Expected 499 marginal levels .002.. .998.')
    crps=torch.zeros_like(residual)
    wis=.5*(residual-q[249]).abs()
    def pin(e,tau):
        return torch.maximum(tau*e,(tau-1)*e)
    previous=pin(residual-q[0],0.)
    for i in range(499):
        tau=(i+1)/500
        current=pin(residual-q[i],tau)
        crps+=(previous+current)*.002
        previous=current
    last=pin(residual-q[-1],1.)
    crps+=(previous+last)*.002
    for i in range(249):
        alpha=(i+1)/500
        low=q[i];high=q[-1-i]
        wis+=alpha*(high-low)+(low-residual).clamp_min(0)+(residual-high).clamp_min(0)
    return crps,wis/249.5


def quantile_pit(sorted_q,residual,chunk=4096):
    """Invert piecewise-linear 499-level quantile CDF; use mid-PIT at atoms."""
    if len(sorted_q)!=499:
        raise ValueError('Expected 499 sorted quantile levels.')
    q=sorted_q.flatten(1);y=residual.flatten()
    out=torch.empty_like(y)
    for i in range(0,y.numel(),chunk):
        j=min(i+chunk,y.numel())
        block=q[:,i:j].transpose(0,1).contiguous()
        yy=y[i:j,None]
        less=(block<yy).sum(1);le=(block<=yy).sum(1)
        left=(less-1).clamp(0,498);right=less.clamp(0,498)
        a=block.gather(1,left[:,None]).squeeze(1)
        b=block.gather(1,right[:,None]).squeeze(1)
        frac=((yy.squeeze(1)-a)/(b-a).clamp_min(1e-12)).clamp(0,1)
        continuous=(less.to(y.dtype)+frac)/500
        continuous=torch.where(less==0,torch.zeros_like(continuous),continuous)
        continuous=torch.where(less==499,torch.ones_like(continuous),continuous)
        jump_lo=torch.where(less==0,torch.zeros_like(continuous),(less+1).to(y.dtype)/500)
        jump_hi=torch.where(le==499,torch.ones_like(continuous),le.to(y.dtype)/500)
        out[i:j]=torch.where(le>less,.5*(jump_lo+jump_hi),continuous)
    return out.reshape_as(residual)
