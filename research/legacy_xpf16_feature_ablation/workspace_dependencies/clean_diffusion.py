"""Conditional diffusion with clean-state parameterization and an observed-state skip."""
import torch
from torch import nn
from models import Diffusion
from global_diffusion import GlobalDenoiser

class CleanDiffusion(Diffusion):
    def __init__(self):
        super().__init__();self.net=GlobalDenoiser()
        nn.init.zeros_(self.net.net[-1].weight);nn.init.zeros_(self.net.net[-1].bias)
    def clean(self,x,t,h):return h[:,:,-1:].expand_as(x)+self.net(x,t,h)
    def training_pair(self,h,y):
        t=torch.randint(100,(len(h),),device=h.device)
        a=self.abar[t].reshape(-1,1,1,1,1)
        return self.clean(a.sqrt()*y+(1-a).sqrt()*torch.randn_like(y),t,h)
    @torch.no_grad()
    def sample(self,history,seed,steps=20):
        g=torch.Generator(device=history.device).manual_seed(seed)
        x=torch.randn(history.shape,device=history.device,generator=g)
        ts=torch.linspace(99,0,steps,device=x.device).long()
        for i,t in enumerate(ts):
            a=self.abar[t];clean=self.clean(x,t.expand(len(x)),history)
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            prev=self.abar[ts[i+1]] if i+1<len(ts) else x.new_tensor(1.)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x
