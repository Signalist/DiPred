"""Per-trajectory squared physical relative L2 objective."""
from rollout_objective import forecast

def relative_loss(pred,y,mean,std):
    error=((pred-y)*std).flatten(1).square().sum(1)
    energy=(y*std+mean).flatten(1).square().sum(1).clamp_min(1e-12)
    return (error/energy).mean()

def loss(model,h,y,mean,std,refine=False):
    return relative_loss(forecast(model,h,refine),y,mean,std)
