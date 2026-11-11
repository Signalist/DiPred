"""Distribution diagnostics for independent paired XPF16 heads.

The 98 model outputs are arranged as 1..49 and 51..99 percent. Q50 is the
arithmetic interpolation of Q49 and Q51. Q0=Q1 and Q100=Q99 are explicitly
assumed. Other intervals are linearly interpolated in probability. Trapezoid
quadrature of pinball loss approximates CRPS; crossings can be sorted first.
"""
import torch


def pinball(y_minus_q, tau):
    return torch.maximum(tau*y_minus_q, (tau-1)*y_minus_q)


def interval_metrics(lower, upper, y, alpha):
    left = (lower-y).clamp_min(0)
    right = (y-upper).clamp_min(0)
    return dict(lower_coverage=(y<=lower).float(), upper_coverage=(y<=upper).float(),
                coverage=((y>=lower)&(y<=upper)).float(), width=upper-lower,
                interval_score=upper-lower+(left+right)/alpha,
                lower_pinball=pinball(y-lower, alpha),
                upper_pinball=pinball(y-upper, 1-alpha),
                lower_miss=(y<lower).float(), upper_miss=(y>upper).float())


def distribution(q98, y, rearrange):
    """Input [98,...], y [...]. Return per-element CRPS, WIS, rank, crossing."""
    assert len(q98)==98
    crossing_before=(q98[1:]<q98[:-1]).float().mean(0)
    q = q98.sort(dim=0).values if rearrange else q98
    difference=q[1:]-q[:-1]
    crossing=(difference<0).float().mean(0)
    crossing_any=(difference<0).any(0).float()
    crossing_magnitude=(-difference).clamp_min(0).mean(0)
    median=.5*(q[48]+q[49])
    q99=torch.cat((q[:49],median.unsqueeze(0),q[49:]),0)
    # Constant tails as specified in module docstring. tau=0 and 1 pinball=0.
    q101=torch.cat((q99[:1],q99,q99[-1:]),0)
    shape=(101,)+(1,)*y.ndim
    tau=torch.arange(101,device=y.device,dtype=y.dtype).reshape(shape)/100
    loss=pinball(y.unsqueeze(0)-q101,tau)
    crps=2*torch.trapz(loss,dx=.01,dim=0)
    wis=.5*(y-median).abs()
    for k in range(1,50):
        alpha=k/100
        lo=q[k-1]
        hi=q[98-k]
        wis=wis+alpha*interval_metrics(lo,hi,y,alpha)['interval_score']
    wis=wis/49.5
    # Invert the piecewise-linear quantile function. On atoms induced by
    # tied quantiles/clamped tails use the midpoint of the CDF jump (mid-PIT).
    pitq=q99 if rearrange else q99.sort(dim=0).values
    less=(pitq<y.unsqueeze(0)).sum(0)
    le=(pitq<=y.unsqueeze(0)).sum(0)
    left_index=(less-1).clamp(0,98).unsqueeze(0)
    right_index=less.clamp(0,98).unsqueeze(0)
    qleft=torch.gather(pitq,0,left_index).squeeze(0)
    qright=torch.gather(pitq,0,right_index).squeeze(0)
    fraction=((y-qleft)/(qright-qleft).clamp_min(1e-12)).clamp(0,1)
    interp=(less.to(y.dtype)+fraction)/100
    interp=torch.where(less==0,torch.zeros_like(interp),interp)
    interp=torch.where(less==99,torch.ones_like(interp),interp)
    jump_lo=torch.where(less==0,torch.zeros_like(interp),(less+1).to(y.dtype)/100)
    jump_hi=torch.where(le==99,torch.ones_like(interp),le.to(y.dtype)/100)
    pit=torch.where(le>less,.5*(jump_lo+jump_hi),interp)
    return dict(crps=crps,wis=wis,pit=pit,crossing=crossing,crossing_any=crossing_any,
                crossing_magnitude=crossing_magnitude,crossing_before=crossing_before)
