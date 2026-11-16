"""Small reference models. Tensor convention: B,C,T,H,W."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Representation(nn.Module):
    def __init__(self, kind='ae', channels=2, latent=8):
        super().__init__()
        self.kind, self.latent = kind, latent
        self.enc = nn.Sequential(nn.Conv2d(channels,32,3,2,1),nn.SiLU(),
            nn.Conv2d(32,latent*(2 if kind=='vae' else 1),3,2,1))
        self.dec = nn.Sequential(nn.ConvTranspose2d(latent,32,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(32,channels,4,2,1))
    def encode(self,x,sample=False):
        b,c,t,h,w=x.shape
        q=self.enc(x.permute(0,2,1,3,4).reshape(b*t,c,h,w))
        kl=x.new_zeros(())
        if self.kind=='vae':
            mu,lv=q.chunk(2,1); lv=lv.clamp(-12,8)
            kl=.5*(mu.square()+lv.exp()-1-lv).mean()
            q=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
        return q.reshape(b,t,self.latent,*q.shape[-2:]).permute(0,2,1,3,4),kl
    def decode(self,z):
        b,c,t,h,w=z.shape
        x=self.dec(z.permute(0,2,1,3,4).reshape(b*t,c,h,w))
        return x.reshape(b,t,2,*x.shape[-2:]).permute(0,2,1,3,4)

class Denoiser(nn.Module):
    def __init__(self,latent=8,width=48):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(2*latent+1,width,3,padding=1),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,latent,3,padding=1))
    def forward(self,x,t,history):
        time=(t.float()/99).view(-1,1,1,1,1).expand(-1,1,*x.shape[2:])
        return self.net(torch.cat([x,history,time],1))

class Diffusion(nn.Module):
    def __init__(self,latent=8):
        super().__init__(); self.net=Denoiser(latent)
        t=torch.linspace(0,1,101)
        a=torch.cos((t+.008)/1.008*math.pi/2).square(); a=a/a[0]
        beta=(1-a[1:]/a[:-1]).clamp(max=.999)
        self.register_buffer('abar',(1-beta).cumprod(0))
    def loss(self,history,future):
        t=torch.randint(100,(future.shape[0],),device=future.device)
        a=self.abar[t].view(-1,1,1,1,1); eps=torch.randn_like(future)
        velocity=a.sqrt()*eps-(1-a).sqrt()*future
        return F.mse_loss(self.net(a.sqrt()*future+(1-a).sqrt()*eps,t,history),velocity)
    @torch.no_grad()
    def sample(self,history,seed,steps=20):
        g=torch.Generator(device=history.device).manual_seed(seed)
        x=torch.randn(history.shape,device=history.device,generator=g)
        ts=torch.linspace(99,0,steps,device=x.device).long()
        for i,t in enumerate(ts):
            a=self.abar[t]; v=self.net(x,t.expand(x.shape[0]),history)
            clean=a.sqrt()*x-(1-a).sqrt()*v
            e=(1-a).sqrt()*x+a.sqrt()*v
            # DDIM eta=0; no clipping of physical/latent states.
            prev=self.abar[ts[i+1]] if i+1<len(ts) else x.new_tensor(1.)
            x=prev.sqrt()*clean+(1-prev).sqrt()*e
        return x

class Corrector(nn.Module):
    """Local non-diagonal spatiotemporal mixing of revealed residuals.

    This is a convolutional transport baseline, not an explicit advection solver.
    """
    def __init__(self,width=24):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(8,width,3,padding=1),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,2,3,padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,h,p,r,m):
        return self.net(torch.cat([h,p,r,m],1))

class OnlineState:
    """Only reveal can update memory; store the actual issued base forecast."""
    def __init__(self,decay=.8): self.decay=decay; self.reset()
    def reset(self): self.pending=None; self.residual=None; self.memory=None
    def reveal(self,truth):
        if self.pending is None: raise RuntimeError('No issued forecast to reveal')
        if truth.shape!=self.pending.shape: raise ValueError('Feedback shape mismatch')
        self.residual=(truth-self.pending).detach().clone()
        self.memory=self.residual.clone() if self.memory is None else self.decay*self.memory+(1-self.decay)*self.residual
        self.pending=None
    def features(self,like):
        z=torch.zeros_like(like)
        return (z if self.residual is None else self.residual,z if self.memory is None else self.memory)
    def issue(self,base):
        if self.pending is not None: raise RuntimeError('Reveal previous window before issuing next')
        self.pending=base.detach().clone()
