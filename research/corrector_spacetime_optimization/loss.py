"""Original consistency objectives and two explicitly separate extensions."""
import torch

from research.corrector_consistency.model import neighbors,MASK_THRESHOLD,COS_FLOOR
from research.corrector_residual_optimization.model import components,LOSSES

NAMES = dict(zip(("DIR","MAG","VEL","ACC","MATCH"),LOSSES))


def extensions(r,y,names):
    values = components(r,y,names)
    if "match2" in names:
        values["match2"] = (r-y).diff(n=2,dim=2).square().mean()
    if "relation" in names:
        mr = torch.linalg.vector_norm(r,dim=1,keepdim=True)
        mt = torch.linalg.vector_norm(y,dim=1,keepdim=True)
        pr = r/mr.clamp_min(COS_FLOOR)
        tr = y/mt.clamp_min(COS_FLOOR)
        valid = mt>MASK_THRESHOLD
        numerator = r.new_zeros(len(r))
        denominator = r.new_zeros(len(r))
        for (a,b),(c,d),(m,n) in zip(neighbors(pr),neighbors(tr),neighbors(valid)):
            mask = (m&n).flatten(1)
            diff = (a*b).sum(1).clamp(-1,1)-(c*d).sum(1).clamp(-1,1)
            numerator += (diff.square().flatten(1)*mask).sum(1)
            denominator += mask.sum(1)
        values["relation"] = (numerator/denominator.clamp_min(1.)).mean()
    return values


def objective(r,y,weights):
    if not weights:
        return (r-y).square().mean()
    values = extensions(r,y,tuple(weights))
    return values["mse"]+sum(values[k]*w for k,w in weights.items())
