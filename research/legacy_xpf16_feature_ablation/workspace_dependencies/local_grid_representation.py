"""Framewise local-grid AE/VAE/RAE;4x16x32 latents,2xcompression."""
import torch
from torch import nn
import torch.nn.functional as F

class LocalEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.input=nn.Sequential(nn.Conv2d(2,32,5,padding=2),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.SiLU());self.res=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1));self.out=nn.Conv2d(64,4,1)
 def forward(self,x):
  q=self.input(x);return self.out(q+self.res(q))

class LocalDecoder(nn.Module):
 def __init__(self):
  super().__init__();self.input=nn.Conv2d(4,64,3,padding=1);self.res=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1));self.out=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,2,3,padding=1))
 def forward(self,z):
  q=self.input(z);q=q+self.res(q);return self.out(F.interpolate(q,scale_factor=2,mode='bilinear',align_corners=False))

class LocalGridRepresentation(nn.Module):
 def __init__(self,kind,pretrained=None):
  super().__init__();assert kind in ['ae','vae','rae','random'];self.kind=kind;self.encoder=LocalEncoder();self.decoder=LocalDecoder()
  if kind=='rae':
   assert pretrained is not None;self.encoder.load_state_dict(pretrained)
  if kind in ['rae','random']:self.encoder.requires_grad_(False)
  if kind=='vae':
   self.mu=nn.Conv2d(4,4,1);self.logvar=nn.Conv2d(4,4,1)
   with torch.no_grad():
    self.mu.weight.zero_();self.mu.weight[:,:,0,0].copy_(torch.eye(4));self.mu.bias.zero_();self.logvar.weight.zero_();self.logvar.bias.fill_(-4.)
 @staticmethod
 def frames(x):
  b,c,t,h,w=x.shape;return x.permute(0,2,1,3,4).reshape(b*t,c,h,w),b,t
 def encode(self,x,sample=False):
  f,b,t=self.frames(x);z=self.encoder(f);kl=x.new_zeros(())
  if self.kind=='vae':
   mu=self.mu(z);lv=self.logvar(z).clamp(-12,8);kl=.5*(mu.square()+lv.exp()-1-lv).mean();z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
  return z.reshape(b,t,4,16,32).permute(0,2,1,3,4),kl
 def decode(self,z):
  f,b,t=self.frames(z);return self.decoder(f).reshape(b,t,2,32,64).permute(0,2,1,3,4)
 def masked_loss(self,x):
  f,_,_=self.frames(x);mask=(torch.rand(len(f),1,8,16,device=x.device)<.75).to(x.dtype);mask=mask.repeat_interleave(4,-2).repeat_interleave(4,-1)
  pred=self.decoder(self.encoder(f*(1-mask)));return ((pred-f).square()*mask).sum()/(mask.sum()*2)
