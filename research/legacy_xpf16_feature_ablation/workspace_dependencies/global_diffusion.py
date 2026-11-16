"""Full-history conditioning for independent 20-frame block diffusion."""
import torch
from torch import nn
from models import Diffusion

class GlobalDenoiser(nn.Module):
    def __init__(self,latent=8,width=48):
        super().__init__()
        self.history=nn.Sequential(nn.Conv2d(latent*20,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1))
        self.net=nn.Sequential(nn.Conv3d(latent*2+32+2,width,3,padding=1),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,latent,3,padding=1))
    def forward(self,x,t,history):
        b,c,nt,h,w=history.shape
        if nt!=20: raise ValueError('Exactly 20 history frames required')
        context=self.history(history.reshape(b,c*nt,h,w))[:,:,None].expand(-1,-1,nt,-1,-1)
        last=history[:,:,-1:].expand_as(x)
        lead=torch.linspace(1/20,1,nt,device=x.device,dtype=x.dtype)[None,None,:,None,None].expand(b,1,nt,h,w)
        noise_time=(t.float()/99).reshape(b,1,1,1,1).expand(b,1,nt,h,w)
        return self.net(torch.cat([x,last,context,lead,noise_time],1))

class GlobalDiffusion(Diffusion):
    def __init__(self):
        super().__init__();self.net=GlobalDenoiser()

@torch.no_grad()
def forecast(rep,diff,h,scale,seed,steps=20,samples=4,anchor=False):
    z=rep.encode(h)[0]
    pred=torch.stack([rep.decode(diff.sample(z/scale,seed+s,steps)*scale) for s in range(samples)]).mean(0)
    if anchor:
        # Offset is computed only from the observed last frame's reconstruction.
        # This repairs decoder bias, not an observed future prediction residual.
        pred=pred+h[:,:,-1:]-rep.decode(z[:,:,-1:])
    return pred
