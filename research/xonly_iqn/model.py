"""X-only PDE quantile forecasting inspired by IQN's cosine embedding."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class XOnlyIQN(nn.Module):
    """Predict any requested marginal quantile of all 20 future frames."""
    def __init__(self, hidden=32, cosine=64):
        super().__init__()
        self.hidden=hidden
        self.cosine=cosine
        self.encoder=nn.Sequential(
            nn.Conv3d(2,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=(1,2,2),dilation=(1,2,2)),
            nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.SiLU())
        self.embedding=nn.Linear(cosine,hidden)
        self.decoder=nn.Sequential(nn.Conv3d(hidden,hidden,1),nn.SiLU(),nn.Conv3d(hidden,2,1))
        self.register_buffer('frequencies',torch.arange(1,cosine+1,dtype=torch.float32)*math.pi)

    def encode(self,history):
        if history.ndim!=5 or history.shape[1:3]!=(2,20):
            raise ValueError('history must have shape [B,2,20,H,W]')
        return self.encoder(history)

    def decode(self,features,history,tau):
        # One tau per trajectory; only observed history determines the center.
        if tau.shape!=(len(history),):raise ValueError('Expected one tau per example')
        embedding=F.relu(self.embedding(torch.cos(tau[:,None]*self.frequencies[None])))
        delta=self.decoder(features*embedding[:,:,None,None,None])
        return history[:,:,-1:].expand_as(delta)+delta

    def forward(self,history,tau):
        return self.decode(self.encode(history),history,tau)


def quantile_huber(prediction,target,tau,kappa=.1):
    residual=target-prediction
    abs_error=residual.abs()
    huber=torch.where(abs_error<=kappa,.5*residual.square(),kappa*(abs_error-.5*kappa))
    weight=(tau[:,None,None,None,None]-(residual.detach()<0).to(residual.dtype)).abs()
    return (weight*huber/kappa).mean()
