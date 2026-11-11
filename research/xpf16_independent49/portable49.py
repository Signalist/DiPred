"""Portable single-/multi-level XPF16 heads with causal two-tail feedback.

Copy this file into the repository's ``reliability`` package and provide
``weights_dir`` and ``offsets_path`` from the independent49 result directory.
Only complete 20-frame blocks may be passed to ``reveal``.
"""
import copy
from pathlib import Path

import torch

from .selected16_predict import RequestedThreePredictor


class XPFIndependentPredictor(RequestedThreePredictor):
    """Freeze DiPred and independently trained 2-output heads."""
    def __init__(self, geometry, weights_dir, offsets_path, levels=None, device='cuda'):
        super().__init__(geometry,method='selected16',device=device)
        self.levels=tuple(range(1,50)) if levels is None else tuple(levels)
        if not self.levels or len(set(self.levels))!=len(self.levels) or any(k<1 or k>49 for k in self.levels):
            raise ValueError('levels must be distinct integers in 1..49')
        self.independent_heads=[]
        for k in self.levels:
            ckpt=torch.load(Path(weights_dir)/f'XPF16-Q{k:02d}.pt',map_location=self.device,weights_only=False)
            head=copy.deepcopy(self.head)
            head.load_state_dict(ckpt['model'],strict=True)
            self.independent_heads.append(head.eval().requires_grad_(False))
        offsets=torch.load(offsets_path,map_location=self.device,weights_only=False)
        self.initial_offsets=offsets[torch.tensor([k-1 for k in self.levels],device=self.device)]
        self.tail_offsets=None
        self.issued_bounds=None

    def reset(self):
        super().reset()
        self.tail_offsets=None
        self.issued_bounds=None

    @torch.no_grad()
    def predict(self, observed):
        captured=[]
        original=self.head.bounds
        def capture(*args):
            captured.append(args)
            return original(*args)
        self.head.bounds=capture
        try:
            result=super().predict(observed)
        finally:
            self.head.bounds=original
        if len(captured)!=1:
            raise RuntimeError('Expected one XPF feature extraction.')
        args=captured[0]
        if self.tail_offsets is None:
            self.tail_offsets=self.initial_offsets[:,None,:,:,None,None,None].expand(-1,len(self.pending),-1,-1,-1,-1,-1).clone()
        raw=[head.bounds(*args) for head in self.independent_heads]
        self.issued_bounds=raw
        base=result['prediction']
        rawlo=[];rawhi=[];online_lo=[];online_hi=[]
        for i,(lo,hi) in enumerate(raw):
            lo2=lo+self.tail_offsets[i,:,0]
            hi2=hi+self.tail_offsets[i,:,1]
            mid=(lo2+hi2)/2
            crossing=lo2>hi2
            lo2=torch.where(crossing,mid,lo2)
            hi2=torch.where(crossing,mid,hi2)
            rawlo.append(base+lo*self.norm['std']);rawhi.append(base+hi*self.norm['std'])
            online_lo.append(base+lo2*self.norm['std']);online_hi.append(base+hi2*self.norm['std'])
        # The parent also computes an older Q05 interval for its own state;
        # do not expose it as if it came from the requested independent heads.
        for key in ('lower','upper','risk_half_width'):
            result.pop(key,None)
        result.update(levels=self.levels, raw_lower=torch.stack(rawlo,dim=1),raw_upper=torch.stack(rawhi,dim=1),
                      calibrated_lower=torch.stack(online_lo,dim=1),calibrated_upper=torch.stack(online_hi,dim=1))
        return result

    @torch.no_grad()
    def reveal(self,truth):
        truth=torch.as_tensor(truth,device=self.device,dtype=torch.float32)
        if self.pending is None or self.issued_bounds is None or truth.shape!=self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError('Feedback must be the complete issued 20-frame target.')
        residual=(truth-self.norm['mean'])/self.norm['std']-self.pending
        for i,(k,(lo,hi)) in enumerate(zip(self.levels,self.issued_bounds)):
            a=k/100
            left=torch.quantile((residual-lo)[...,::4,::4].flatten(2),a,dim=2)
            right=torch.quantile((residual-hi)[...,::4,::4].flatten(2),1-a,dim=2)
            estimate=torch.stack((left,right),1)[...,None,None,None]
            self.tail_offsets[i]=.5*self.tail_offsets[i]+.5*estimate
        self.issued_bounds=None
        super().reveal(truth)
