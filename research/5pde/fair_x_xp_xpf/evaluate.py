import torch
from calibration_grid import per_level_quantiles
LEVELS = torch.arange(1, 500, dtype=torch.float64) / 500
ALPHA = LEVELS[:249]
def _add_at(acc, indices, values):
    acc.scatter_add_(0, indices.reshape(-1), values.reshape(-1).to(torch.float64))

def pair_quantiles(blocks, levels):
    value = torch.cat(blocks, dim=1).permute(0, 2, 1, 3, 4, 5).flatten(2)
    result = []
    for i in range(0, 249, 8):
        result.append(per_level_quantiles(value[i:i+8].cuda(), levels[i:i+8].cuda()).cpu())
    return torch.cat(result)

def sorted_cdf_sum(q, probs, grid, chunk=8192):
    """CDF sum for already rearranged quantiles; uses endpoint atoms."""
    n=len(grid)
    slope=torch.zeros(n+1,device=q.device,dtype=torch.float64)
    intercept=torch.zeros_like(slope)
    probs=probs.to(device=q.device,dtype=torch.float64)
    for start in range(0,q.shape[1],chunk):
        values=q[:,start:start+chunk].to(torch.float64)
        left,right=values[:-1],values[1:]
        weights=(probs[1:]-probs[:-1])[:,None].expand_as(left)
        equal=right==left
        local=torch.where(equal,torch.zeros_like(left),weights/(right-left).clamp_min(1e-300))
        a,b=torch.bucketize(left.contiguous(),grid),torch.bucketize(right.contiguous(),grid)
        _add_at(slope,a,local);_add_at(slope,b,-local)
        _add_at(intercept,a,-local*left+weights*equal);_add_at(intercept,b,local*right)
        ones=torch.ones(values.shape[1],device=q.device,dtype=torch.float64)
        _add_at(intercept,torch.bucketize(values[0].contiguous(),grid),ones*probs[0])
        _add_at(intercept,torch.bucketize(values[-1].contiguous(),grid),ones*(1-probs[-1]))
    return slope[:-1].cumsum(0)*grid+intercept[:-1].cumsum(0)

def quantile_crps(q, y, levels):
    """Trapezoidal 2*pinball integral with endpoint-clamped quantiles."""
    total = .5 * ((q[0]-y).clamp_min(0) + (y-q[-1]).clamp_min(0))
    for i in range(0, len(levels), 16):
        error = y[None] - q[i:i+16]
        t = levels[i:i+16, None, None, None, None, None]
        total = total + torch.maximum(t*error, (t-1)*error).sum(0)
    return total * .004

def bounds(q, cal, controller, median_controller):
    lo, hi = q[:249], q[250:].flip(0)
    c = cal['fixed'][:, None, :, None, None, None]
    a, b = lo - c, hi + c
    invalid = a > b
    mid = (a + b) / 2
    a, b = torch.where(invalid, mid, a), torch.where(invalid, mid, b)
    fixed = torch.cat((a, q[249:250], b.flip(0)))
    online_invalid = (lo + controller.offsets[:,:,0]) > (hi + controller.offsets[:,:,1])
    online_projected = float(online_invalid.float().mean())
    a, b = controller.issue(lo, hi)
    center = median_controller.issue(q[249])
    online = torch.cat((a, center[None], b.flip(0)))
    return (q, fixed, online), (0., float(invalid.float().mean()), online_projected)
