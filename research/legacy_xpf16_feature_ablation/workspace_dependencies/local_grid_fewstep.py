"""Native four-level cosine DDPM in learned PDE grid latents;no distillation."""
import math
import torch
from local_grid_diffusion import LocalGridDiffusion

class LocalGridFewstep(LocalGridDiffusion):
 def __init__(self):
  super().__init__();t=torch.arange(5,dtype=torch.float64)/4;a=torch.cos((t+.008)/1.008*math.pi/2).square();a=a/a[0];betas=1-a[1:]/a[:-1];betas[-1]=1.;self.register_buffer('betas',betas.float());self.abar=torch.cumprod(1-self.betas,0)
 def clean(self,x,t,h):
  a=self.abar[t,None,None,None,None];return a.sqrt()*x-(1-a).sqrt()*self.net(x,t*33,h)
 def loss(self,h,y):
  lead=int(torch.randint(4,(),device=h.device))*5;seq=torch.cat([h,y],2);h=seq[:,:,lead:lead+20];y=seq[:,:,lead+20:lead+25];t=torch.randint(4,(len(h),),device=h.device);a=self.abar[t,None,None,None,None];eps=torch.randn_like(y);target=a.sqrt()*eps-(1-a).sqrt()*y
  return (self.net(a.sqrt()*y+(1-a).sqrt()*eps,t*33,h)-target).square().mean()
 def sample_grad(self,h,seed,sign=1):
  g=torch.Generator(device=h.device).manual_seed(seed);shape=(len(h),4,5,16,32);x=torch.randn(shape,device=h.device,dtype=h.dtype,generator=g)*sign
  for t in reversed(range(4)):
   a=self.abar[t];prev=self.abar[t-1] if t else x.new_tensor(1.);beta=self.betas[t];clean=self.clean(x,torch.full((len(h),),t,device=h.device,dtype=torch.long),h);mean=prev.sqrt()*beta/(1-a)*clean+(1-beta).sqrt()*(1-prev)/(1-a)*x;var=beta*(1-prev)/(1-a)
   x=mean+var.sqrt()*torch.randn(shape,device=h.device,dtype=h.dtype,generator=g)*sign if t else clean
  return x
 @torch.no_grad()
 def sample(self,h,seed,sign=1):return self.sample_grad(h,seed,sign)
 def rollout_grad(self,h,seed,sign=1):
  blocks=[]
  for lead in range(4):
   z=self.sample_grad(h,seed+lead*1009,sign);blocks.append(z);h=torch.cat([h[:,:,5:],z],2)
  return torch.cat(blocks,2)
