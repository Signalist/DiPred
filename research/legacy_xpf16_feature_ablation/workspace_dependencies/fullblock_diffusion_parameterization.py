"""Matched physical-space U-Net for independent 20-to-20 forecasting."""
import torch
from torch import nn
import torch.nn.functional as F
from clean_diffusion import CleanDiffusion

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.a=nn.Conv2d(cin,cout,3,padding=1);self.b=nn.Conv2d(cout,cout,3,padding=1)
        self.norm1=nn.GroupNorm(8,cout);self.norm2=nn.GroupNorm(8,cout);self.time=nn.Linear(32,cout)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x,t):
        y=F.silu(self.norm1(self.a(x))+self.time(t)[:,:,None,None])
        return self.skip(x)+self.b(F.silu(self.norm2(y)))

class BlockNet(nn.Module):
    def __init__(self,channels=(48,96,176)):
        super().__init__();self.time=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,32))
        a,b,c=channels;self.first=ResidualBlock(80,a);self.down1=nn.Conv2d(a,b,4,2,1);self.enc1=ResidualBlock(b,b)
        self.down2=nn.Conv2d(b,c,4,2,1);self.mid=ResidualBlock(c,c)
        self.up1=ResidualBlock(c+b,b);self.up2=ResidualBlock(b+a,a);self.out=nn.Conv2d(a,40,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,t,h):
        b,c,nt,ny,nx=h.shape
        if (c,nt)!=(2,20) or h.shape!=x.shape:raise ValueError('Expected equal B,2,20,H,W blocks')
        u=t.to(h.dtype)/99;te=self.time(torch.stack([u,u*u,torch.sin(torch.pi*u),torch.cos(torch.pi*u)],1))
        f=self.first(torch.cat([h.reshape(b,40,ny,nx),x.reshape(b,40,ny,nx)],1),te)
        g=self.enc1(self.down1(f),te);z=self.mid(self.down2(g),te)
        z=self.up1(torch.cat([F.interpolate(z,size=g.shape[-2:],mode='bilinear',align_corners=False),g],1),te)
        z=self.up2(torch.cat([F.interpolate(z,size=f.shape[-2:],mode='bilinear',align_corners=False),f],1),te)
        return self.out(z).reshape_as(h)

class BlockDiffusion(nn.Module):
    """Full-state VP diffusion; matched raw x0/v heads, no CNN warm start."""
    def __init__(self,parameterization):
        super().__init__()
        if parameterization not in ['x0','v']: raise ValueError(parameterization)
        self.parameterization=parameterization
        self.net=BlockNet()
        from models import Diffusion
        self.register_buffer('abar',Diffusion().abar.clone())

    def clean(self,x,t,h):
        out=self.net(x,t,h)
        a=self.abar[t].reshape(-1,1,1,1,1)
        return out if self.parameterization=='x0' else a.sqrt()*x-(1-a).sqrt()*out

    def loss(self,h,y):
        t=torch.randint(100,(len(h),),device=h.device)
        a=self.abar[t].reshape(-1,1,1,1,1)
        eps=torch.randn_like(y)
        x=a.sqrt()*y+(1-a).sqrt()*eps
        target=y if self.parameterization=='x0' else a.sqrt()*eps-(1-a).sqrt()*y
        return F.mse_loss(self.net(x,t,h),target)

    @torch.no_grad()
    def sample(self,h,seed,steps=20):
        g=torch.Generator(device=h.device).manual_seed(seed)
        x=torch.randn(h.shape,device=h.device,generator=g)
        ts=torch.linspace(99,0,steps,device=h.device).long()
        for i,t in enumerate(ts):
            a=self.abar[t]
            clean=self.clean(x,t.expand(len(h)),h)
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            prev=self.abar[ts[i+1]] if i+1<len(ts) else x.new_tensor(1.)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x

    @torch.no_grad()
    def forecast(self,h):
        return torch.stack([self.sample(h,100042+i) for i in range(4)]).mean(0)
