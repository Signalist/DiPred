"""Independent XPF16 heads on the 0.2%-probability grid, causal feedback.

Weights must first be extracted into ``weights_dir`` from the published
archive. Default output covers all 499 quantiles; a subset of interval IDs may
be selected to reduce latency and memory. Every head has distinct weights.
"""
import copy
from pathlib import Path

import torch

from .selected16_predict import RequestedThreePredictor
from .calibration_grid02 import GridCalibrator,MedianCalibrator


RESULTS=Path(__file__).resolve().parent.parent/'results/reliability/xpf16_02grid_2048'


class Grid02Predictor(RequestedThreePredictor):
    """Return original-scale quantiles, updating only after complete feedback."""
    def __init__(self,geometry,weights_dir,calibration=None,tail_ids=None,device='cuda'):
        super().__init__(geometry,method='selected16',device=device)
        self.tail_ids=tuple(range(1,250)) if tail_ids is None else tuple(sorted(tail_ids))
        if not self.tail_ids or len(set(self.tail_ids))!=len(self.tail_ids) or any(k<1 or k>249 for k in self.tail_ids):
            raise ValueError('tail_ids must contain distinct integers in 1..249')
        self.levels=tuple([k/500 for k in self.tail_ids]+[.5]+[1-k/500 for k in reversed(self.tail_ids)])
        self.grid_heads=[]
        for k in (*self.tail_ids,250):
            packed=torch.load(Path(weights_dir)/f'XPF16-P{k:03d}.pt',map_location=self.device,weights_only=False)
            if abs(packed['config']['lower']-k/500)>1e-12:
                raise ValueError('Checkpoint does not match the requested probability.')
            head=copy.deepcopy(self.head);head.load_state_dict(packed['model'],strict=True)
            self.grid_heads.append(head.eval().requires_grad_(False))
        path=Path(calibration) if calibration else RESULTS/'calibration.pt'
        states=torch.load(path,map_location=self.device,weights_only=False)
        select=torch.tensor([k-1 for k in self.tail_ids],device=self.device)
        self.grid_initial=states['xpf'].index_select(0,select)
        self.median_initial=states['xpf_median']
        self.grid_controller=None
        self.median_controller=None

    def reset(self):
        super().reset()
        self.grid_controller=None
        self.median_controller=None

    @torch.no_grad()
    def predict(self,observed):
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
            raise RuntimeError('Expected one shared XPF16 feature construction.')
        pairs=[head.bounds(*captured[0]) for head in self.grid_heads]
        lo=torch.stack([x[0] for x in pairs[:-1]])
        hi=torch.stack([x[1] for x in pairs[:-1]])
        median=.5*(pairs[-1][0]+pairs[-1][1])
        if self.grid_controller is None:
            levels=torch.tensor([k/500 for k in self.tail_ids],device=self.device,dtype=torch.float64)
            self.grid_controller=GridCalibrator(levels,self.grid_initial,len(self.pending))
            self.median_controller=MedianCalibrator(self.median_initial,len(self.pending))
        a,b=self.grid_controller.issue(lo,hi)
        m=self.median_controller.issue(median)
        raw=torch.cat((lo,median.unsqueeze(0),hi.flip(0)))
        online=torch.cat((a,m.unsqueeze(0),b.flip(0)))
        point=result['prediction'].unsqueeze(0)
        for key in ('lower','upper','risk_half_width'):result.pop(key,None)
        # [B,Q,2,20,32,64], quantile levels are in ascending probability order.
        result.update(levels=self.levels,raw_quantiles=(point+raw*self.norm['std']).transpose(0,1),
                      calibrated_quantiles=(point+online*self.norm['std']).transpose(0,1))
        return result

    @torch.no_grad()
    def reveal(self,truth):
        truth=torch.as_tensor(truth,device=self.device,dtype=torch.float32)
        if self.pending is None or self.grid_controller is None or truth.shape!=self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError('Feedback must contain all 20 finite frames of the issued window.')
        residual=(truth-self.norm['mean'])/self.norm['std']-self.pending
        self.grid_controller.reveal(residual)
        self.median_controller.reveal(residual)
        super().reveal(truth)
