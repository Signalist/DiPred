"""Full-state conditional diffusion in a frozen local4channel grid codec."""
import torch
from torch import nn
import torch.nn.functional as F
from fiveblock_diffusion import BlockNet,ResidualBlock

class LocalGridNet(BlockNet):
 def __init__(self):
  super().__init__();self.first=ResidualBlock(100,48);self.out=nn.Conv2d(48,20,3,padding=1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x,t,h):
  b,c,nt,ny,nx=h.shape;assert (c,nt)==(4,20) and x.shape==(b,4,5,ny,nx)
  u=t.to(h.dtype)/99;te=self.time(torch.stack([u,u*u,torch.sin(torch.pi*u),torch.cos(torch.pi*u)],1));f=self.first(torch.cat([h.reshape(b,80,ny,nx),x.reshape(b,20,ny,nx)],1),te);g=self.enc1(self.down1(f),te);z=self.mid(self.down2(g),te);z=self.up1(torch.cat([F.interpolate(z,size=g.shape[-2:],mode='bilinear',align_corners=False),g],1),te);z=self.up2(torch.cat([F.interpolate(z,size=f.shape[-2:],mode='bilinear',align_corners=False),f],1),te)
  return self.out(z).reshape(b,4,5,ny,nx)

class LocalGridDiffusion(nn.Module):
 def __init__(self):
  super().__init__();self.net=LocalGridNet();t=torch.linspace(0,1,101,dtype=torch.float64);a=torch.cos((t+.008)/1.008*torch.pi/2).square();a=a/a[0];self.register_buffer('abar',torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0).float())
 def loss(self,h,y):
  lead=int(torch.randint(4,(),device=h.device))*5;seq=torch.cat([h,y],2);h=seq[:,:,lead:lead+20];y=seq[:,:,lead+20:lead+25];t=torch.randint(100,(len(h),),device=h.device);a=self.abar[t,None,None,None,None];eps=torch.randn_like(y);target=a.sqrt()*eps-(1-a).sqrt()*y
  return (self.net(a.sqrt()*y+(1-a).sqrt()*eps,t,h)-target).square().mean()
 @torch.no_grad()
 def sample(self,h,seed,sign=1):
  g=torch.Generator(device=h.device).manual_seed(seed);x=torch.randn((len(h),4,5,16,32),device=h.device,generator=g)*sign;ts=torch.linspace(99,0,20,device=h.device).long()
  for i,t in enumerate(ts):
   a=self.abar[t];v=self.net(x,t.expand(len(h)),h);clean=a.sqrt()*x-(1-a).sqrt()*v;eps=(1-a).sqrt()*x+a.sqrt()*v;prev=self.abar[ts[i+1]] if i+1<len(ts) else x.new_tensor(1.);x=prev.sqrt()*clean+(1-prev).sqrt()*eps
  return x
 @torch.no_grad()
 def rollout(self,h,seed,sign=1):
  pieces=[]
  for block in range(4):
   z=self.sample(h,seed+block*1009,sign);pieces.append(z);h=torch.cat([h[:,:,5:],z],2)
  return torch.cat(pieces,2)
