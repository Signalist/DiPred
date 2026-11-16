"""Native latent v risk plus actual-field full20finite-ensemble mean risk."""
import torch
from local_grid_fewstep import LocalGridFewstep
from relative_rollout_objective import relative_loss

class MeanRiskDiffusion(LocalGridFewstep):
 def sample_grad(self,h,seed,sign=1):
  g=torch.Generator(device=h.device).manual_seed(seed);x=torch.randn((len(h),4,5,16,32),device=h.device,generator=g)*sign
  for t in reversed(range(4)):
   a=self.abar[t];v=self.net(x,torch.full((len(h),),t*33,device=h.device,dtype=torch.long),h);clean=a.sqrt()*x-(1-a).sqrt()*v;eps=(1-a).sqrt()*x+a.sqrt()*v;prev=self.abar[t-1] if t else x.new_tensor(1.);x=prev.sqrt()*clean+(1-prev).sqrt()*eps
  return x

def point_risk(m,rep,center,scale,h,y,mean,std,seeds):
 a=rep.decode(m.rollout_grad(h,int(seeds[0]))*scale+center);b=rep.decode(m.rollout_grad(h,int(seeds[1]))*scale+center)
 return relative_loss((a+b)/2,y,mean,std)

def objective(m,rep,center,scale,h,z,y,mean,std,weight):
 native=m.loss(h,z);seeds=torch.randint(1000000,2000000,(2,),device=h.device)
 if weight==0:return native,dict(native=float(native),point_risk=0.)
 risk=point_risk(m,rep,center,scale,h,y,mean,std,seeds)
 return native+weight*risk,dict(native=float(native),point_risk=float(risk))
