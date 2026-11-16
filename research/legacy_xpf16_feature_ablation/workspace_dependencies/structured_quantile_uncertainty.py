"""Richer history conditioning and train-only spatial error normalization."""

import torch
from torch import nn
import torch.nn.functional as F


MODES = ("rich", "template", "template_base")


class StructuredQuantileHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.history_projection = nn.Conv2d(40, 80, 1)
        self.feature_projection = nn.Conv2d(48, 160, 1)
        self.net = nn.Sequential(
            nn.Conv3d(28, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1,2,2), dilation=(1,2,2)),
            nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv3d(24, 4, 1))

    def bounds(self, base, last, history, scale):
        b, _, _, h, w = base.shape
        historical = self.history_projection(history.reshape(b,40,h,w)).reshape(b,4,20,h,w)
        features = self.feature_projection(last).reshape(b,8,20,h,w)
        if self.mode.endswith("base"):
            features = features * 0
        yy = torch.linspace(-1,1,h,device=base.device,dtype=base.dtype)
        xx = torch.linspace(-1,1,w,device=base.device,dtype=base.dtype)
        y,x = torch.meshgrid(yy,xx,indexing="ij")
        coordinates = torch.stack([x,y])[None,:,None].expand(b,2,20,h,w)
        value = self.net(torch.cat([base,historical,features,coordinates],1))
        mid, width = value[:,:2], F.softplus(value[:,2:])+1e-4
        return (mid-width)*scale, (mid+width)*scale

    def loss(self, base, last, history, residual, scale):
        lower,upper = self.bounds(base,last,history,scale)
        # A train-only fixed position/lead scale preserves each conditional
        # quantile target, while balancing optimization across spatial regions.
        e_lower,e_upper = (residual-lower)/scale,(residual-upper)/scale
        return (torch.maximum(.05*e_lower,-.95*e_lower)
                + torch.maximum(.95*e_upper,-.05*e_upper)).mean()
