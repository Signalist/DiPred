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
        a,b,c=channels;self.first=ResidualBlock(50,a);self.down1=nn.Conv2d(a,b,4,2,1);self.enc1=ResidualBlock(b,b)
        self.down2=nn.Conv2d(b,c,4,2,1);self.mid=ResidualBlock(c,c)
        self.up1=ResidualBlock(c+b,b);self.up2=ResidualBlock(b+a,a);self.out=nn.Conv2d(a,10,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,t,h):
        b,c,nt,ny,nx=h.shape
        if (c,nt)!=(2,20) or x.shape!=(b,2,5,ny,nx):raise ValueError('Expected history20 and future5')
        u=t.to(h.dtype)/99;te=self.time(torch.stack([u,u*u,torch.sin(torch.pi*u),torch.cos(torch.pi*u)],1))
        f=self.first(torch.cat([h.reshape(b,40,ny,nx),x.reshape(b,10,ny,nx)],1),te)
        g=self.enc1(self.down1(f),te);z=self.mid(self.down2(g),te)
        z=self.up1(torch.cat([F.interpolate(z,size=g.shape[-2:],mode='bilinear',align_corners=False),g],1),te)
        z=self.up2(torch.cat([F.interpolate(z,size=f.shape[-2:],mode='bilinear',align_corners=False),f],1),te)
        return self.out(z).reshape(b,2,5,ny,nx)

from fullblock_diffusion_parameterization import BlockDiffusion as Base

class BlockDiffusion(Base):
    def __init__(self,parameterization):
        super().__init__(parameterization)
        self.net=BlockNet()

    def loss(self,h,y):
        # Teacher-forced subblock pairs at training only. Forecast has no target.
        lead=int(torch.randint(4,(),device=h.device))*5
        seq=torch.cat([h,y],2)
        return super().loss(seq[:,:,lead:lead+20],seq[:,:,lead+20:lead+25])

    @torch.no_grad()
    def sample(self,h,seed,steps=20):
        g=torch.Generator(device=h.device).manual_seed(seed)
        x=torch.randn((len(h),2,5,*h.shape[-2:]),device=h.device,generator=g)
        ts=torch.linspace(99,0,steps,device=h.device).long()
        for i,t in enumerate(ts):
            a=self.abar[t];clean=self.clean(x,t.expand(len(h)),h)
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            prev=self.abar[ts[i+1]] if i+1<len(ts) else x.new_tensor(1.)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x

    @torch.no_grad()
    def forecast(self,h):
        trajectories=[]
        for sample in range(4):
            history=h;blocks=[]
            for lead in range(4):
                p=self.sample(history,100042+sample+100*lead)
                blocks.append(p);history=torch.cat([history[:,:,5:],p],2)
            trajectories.append(torch.cat(blocks,2))
        return torch.stack(trajectories).mean(0)
