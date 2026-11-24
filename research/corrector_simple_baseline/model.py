"""The additive XP architecture with a residual-only forward interface."""
import torch
from torch import nn


class SimpleCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4,8,3,padding=1),
            nn.SiLU(),
            nn.Conv3d(8,8,3,padding=1),
            nn.SiLU(),
            nn.Conv3d(8,2,3,padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,p):
        assert x.shape==p.shape and x.shape[1:3]==(2,20)
        return self.net(torch.cat((x,p),dim=1).contiguous(memory_format=torch.channels_last_3d))


def network(device):
    return SimpleCorrector().to(device=device,memory_format=torch.channels_last_3d)


def fields(split,norm):
    return tuple((v-norm["mean"])/norm["std"] for v in (split.x,split.p))


@torch.inference_mode()
def physical_residual(net,inputs,scale):
    net.eval()
    return torch.cat([net(x,p)*scale for x,p in zip(*(v.split(8) for v in inputs))])
