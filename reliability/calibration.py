"""Empirical calibration after complete20-frame feedback; no CP guarantee."""

from collections import deque
import math
import torch


FEATURE_NAMES = ("log_width", "log_gradient", "log_departure", "log_history_variability",
                 "log_forecast_change", "forecast_magnitude", "lead", "lead_squared", "x", "y", "xy")


def grouped_quantile(ratio, grouping, probability=.9, pooled=False):
    """Quantiles within a trajectory, or pooled across calibration trajectories.

    Return broadcastable B,C,T,H,W scale arrays. Each tile is eight by eight.
    Pixel scores are correlated; these empirical quantiles have no CP guarantee.
    """
    b, c, t, h, w = ratio.shape
    tiles = ratio.reshape(b, c, t, 4, 8, 8, 8)
    if grouping == "global":
        values = ratio.permute(1, 0, 2, 3, 4).reshape(c, -1) if pooled else ratio.flatten(2)
        q = torch.quantile(values, probability, dim=-1)
        return q.reshape(1 if pooled else b, c, 1, 1, 1)
    if grouping == "lead":
        values = ratio.permute(1, 2, 0, 3, 4).reshape(c, t, -1) if pooled else ratio.flatten(3)
        q = torch.quantile(values, probability, dim=-1)
        return q.reshape(1 if pooled else b, c, t, 1, 1)
    if grouping == "patch":
        values = tiles.permute(1, 3, 5, 0, 2, 4, 6).reshape(c, 4, 8, -1) if pooled else tiles.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, c, 4, 8, -1)
        q = torch.quantile(values, probability, dim=-1).reshape(1 if pooled else b, c, 1, 4, 8)
    else:
        assert grouping == "lead_patch"
        values = tiles.permute(1, 2, 3, 5, 0, 4, 6).reshape(c, t, 4, 8, -1) if pooled else tiles.permute(0, 1, 2, 3, 5, 4, 6).flatten(5)
        q = torch.quantile(values, probability, dim=-1).reshape(1 if pooled else b, c, t, 4, 8)
    return q.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)


def shape_from_scores(ratio,mode,pooled=False):
    global_q=grouped_quantile(ratio,"global",pooled=pooled).clamp(1e-6,1e3)
    if mode.startswith("global"):
        return torch.ones_like(global_q)
    lead=grouped_quantile(ratio,"lead",pooled=pooled).clamp(1e-6,1e3)/global_q
    if mode.startswith("separable"):
        patch=grouped_quantile(ratio,"patch",pooled=pooled).clamp(1e-6,1e3)/global_q
        lead=lead*patch.sqrt()
    return lead/lead.log().mean((2,3,4),keepdim=True).exp()


class ConditionalRiskCalibrator:
    def __init__(self,mode,initial,batch):
        self.mode=mode
        self.shape=initial["shape"].expand(batch,*initial["shape"].shape[1:]).clone()
        self.amplitude=initial["amplitude"].expand(batch,2,1,1,1).clone()
        self.bias=torch.zeros_like(self.amplitude)
        self.beta=self.amplitude.new_zeros(batch,2,len(FEATURE_NAMES))
        self.pending=None
        self.revealed_windows=0

    def conditional_shape(self,features):
        log_shape=(features*self.beta[:,:,:,None,None,None]).sum(2)
        return log_shape.clamp(-math.log(2),math.log(2)).exp()

    def issue(self,lower,upper,features):
        if self.pending is not None:
            raise RuntimeError("Reveal the whole preceding20-frame window first.")
        if lower.shape!=upper.shape or lower.shape[1:]!=(2,20,32,64):
            raise ValueError("Expected complete[B,2,20,32,64] error bounds.")
        if self.mode=="ridge_static":
            features=features.clone()
            features[:,:,1:6]=0
        mid,width=.5*(lower+upper),(.5*(upper-lower)).clamp_min(1e-6)
        radius=width*self.conditional_shape(features)*self.shape*self.amplitude*self.bias.exp()
        lo,hi=mid-radius,mid+radius
        self.pending=(mid.clone(),width.clone(),lo.clone(),hi.clone(),features.clone())
        return lo,hi

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape:
            raise RuntimeError("Feedback requires the entire previously issued20-frame block.")
        mid,width,lo,hi,features=self.pending
        error=(residual-mid).abs()
        ratio=error/width
        if self.mode!="ridge_off":
            # Condition on the current known field; fit only AFTER truth reveal.
            target=((error+1e-4)/width/self.shape).log().clamp(-8,8).flatten(2)
            target=target-target.mean(-1,keepdim=True)
            x=features.flatten(3).reshape(-1,len(FEATURE_NAMES),features[0,0,0].numel())
            y=target.reshape(-1,target.shape[-1],1)
            count=x.shape[-1]
            gram=torch.bmm(x,x.transpose(1,2))/count
            cross=torch.bmm(x,y)/count
            identity=torch.eye(len(FEATURE_NAMES),device=x.device,dtype=x.dtype)[None]
            estimate=torch.linalg.solve(gram+.25*identity,cross).squeeze(-1).reshape_as(self.beta)
            self.beta=(.5*self.beta+.5*estimate.clamp(-1,1))
        conditional=self.conditional_shape(features)
        adjusted=ratio/conditional
        fit_shape=shape_from_scores(adjusted,"separable_integral")
        self.shape=(self.shape*fit_shape).sqrt()
        self.shape=self.shape/self.shape.log().mean((2,3,4),keepdim=True).exp()
        self.amplitude=grouped_quantile(adjusted/self.shape,"global").clamp(1e-6,1e3)
        coverage=((residual>=lo)&(residual<=hi)).float().mean((2,3,4),keepdim=True)
        self.bias=(self.bias+.5*(.9-coverage)).clamp(-math.log(2),math.log(2))
        self.pending=None
        self.revealed_windows+=1


class MomentRiskCalibrator(ConditionalRiskCalibrator):
    """A fixed five-window effective memory, without retaining field targets.

    Compared with per-window ridge, only the Gram/cross moments are pooled.
    The ridge penalty, beta EMA, feature bounds and shape rule stay unchanged.
    All moment updates happen after complete-window feedback.
    """

    def __init__(self, mode, initial, batch):
        super().__init__("ridge_dynamic", initial, batch)
        self.moment_mode = mode
        self.gram = None
        self.cross = None

    def reveal(self, residual):
        if self.pending is None or residual.shape != self.pending[0].shape:
            raise RuntimeError("Moment updates require the complete issued 20-frame window.")
        mid, width, lower, upper, features = self.pending
        error = (residual - mid).abs()
        target = ((error + 1e-4) / width / self.shape).log().clamp(-8, 8).flatten(2)
        target = target - target.mean(-1, keepdim=True)
        count = features[0, 0, 0].numel()
        x = features.flatten(3).reshape(-1, len(FEATURE_NAMES), count)
        y = target.reshape(-1, count, 1)
        gram = torch.bmm(x, x.transpose(1, 2)) / count
        cross = torch.bmm(x, y) / count
        if self.gram is None:
            self.gram, self.cross = gram, cross
        else:
            self.gram = 0.8 * self.gram + 0.2 * gram
            self.cross = 0.8 * self.cross + 0.2 * cross
        identity = torch.eye(len(FEATURE_NAMES), device=x.device, dtype=x.dtype)[None]
        estimate = torch.linalg.solve(self.gram + 0.25 * identity, self.cross)
        estimate = estimate.squeeze(-1).reshape_as(self.beta).clamp(-1, 1)
        self.beta = 0.5 * self.beta + 0.5 * estimate

        # Preserve the previous controller's scale and coverage rules exactly.
        adjusted = error / width / self.conditional_shape(features)
        fitted_shape = shape_from_scores(adjusted, "separable_integral")
        self.shape = (self.shape * fitted_shape).sqrt()
        self.shape = self.shape / self.shape.log().mean((2, 3, 4), keepdim=True).exp()
        self.amplitude = grouped_quantile(adjusted / self.shape, "global").clamp(1e-6, 1e3)
        coverage = ((residual >= lower) & (residual <= upper)).float().mean((2, 3, 4), keepdim=True)
        gain = 2.0 if self.moment_mode == "moment_fastcal" else 0.5
        self.bias = (self.bias + gain * (0.9 - coverage)).clamp(-math.log(2), math.log(2))
        self.pending = None
        self.revealed_windows += 1


class SmoothedScaleCalibrator(MomentRiskCalibrator):
    """Use a log-scale EMA after the first observed trajectory window.

    The first feedback sets a trajectory-specific amplitude directly. Later
    amplitudes use the geometric mean of the previous and newly fitted scales.
    All fitting still follows the full 20-frame feedback, never a partial block.
    """

    def __init__(self, mode, initial, batch):
        base_mode = "moment_fastcal" if mode == "scale_fastcal" else "moment_dynamic"
        super().__init__(base_mode, initial, batch)

    def reveal(self, residual):
        old_amplitude = self.amplitude.clone()
        completed = self.revealed_windows
        super().reveal(residual)
        if completed > 0:
            self.amplitude = (old_amplitude * self.amplitude).sqrt()


def fit_memory_quantile(memory, start):
    """Average blockwise pinball gradients, profiling an intercept per block.

    Each stored target uses that block's pre-feedback separable shape. No
    future labels are available, and different blocks keep separate intercepts
    so that global magnitude changes do not directly dictate risk slopes.
    """
    blocks = []
    for features, target in memory:
        x = features.flatten(3).reshape(-1, features.shape[2], features[0, 0, 0].numel())
        y = target.flatten(2).reshape(-1, target[0, 0].numel())
        blocks.append((x, y))
    beta = start.reshape(-1, start.shape[-1]).clone()
    for iteration in range(12):
        gradient = torch.zeros_like(beta)
        for x, y in blocks:
            prediction = torch.bmm(beta[:, None], x).squeeze(1)
            intercept = torch.quantile(y - prediction, 0.9, dim=-1, keepdim=True)
            below = (y < prediction + intercept).to(x.dtype)
            gradient += torch.bmm(x, (below - 0.9)[:, :, None]).squeeze(-1) / x.shape[-1]
        gradient = gradient / len(blocks) + 0.025 * beta
        beta = (beta - 0.5 / math.sqrt(iteration + 1) * gradient).clamp(-1, 1)
    return beta.reshape_as(start)


class MemoryQuantileCalibrator(ConditionalRiskCalibrator):
    def __init__(self, mode, initial, batch):
        super().__init__("ridge_dynamic", initial, batch)
        self.memory_mode = mode
        self.memory = deque(maxlen=3)

    def reveal(self, residual):
        if self.pending is None or residual.shape != self.pending[0].shape:
            raise RuntimeError("Memory accepts only the complete issued 20-frame block.")
        mid, width, lower, upper, features = self.pending
        error = (residual - mid).abs()
        target = ((error + 1e-4) / width / self.shape).log().clamp(-8, 8)
        self.memory.append((features.clone(), target.clone()))
        estimate = fit_memory_quantile(self.memory, self.beta)
        self.beta = 0.5 * self.beta + 0.5 * estimate
        adjusted = error / width / self.conditional_shape(features)
        fitted_shape = shape_from_scores(adjusted, "separable_integral")
        self.shape = (self.shape * fitted_shape).sqrt()
        self.shape = self.shape / self.shape.log().mean((2, 3, 4), keepdim=True).exp()
        fitted_amplitude = grouped_quantile(adjusted / self.shape, "global").clamp(1e-6, 1e3)
        if self.memory_mode == "memory_scale_fast" and self.revealed_windows > 0:
            self.amplitude = (self.amplitude * fitted_amplitude).sqrt()
        else:
            self.amplitude = fitted_amplitude
        coverage = ((residual >= lower) & (residual <= upper)).float().mean((2, 3, 4), keepdim=True)
        gain = 0.5 if self.memory_mode == "memory_dynamic" else 2.0
        self.bias = (self.bias + gain * (0.9 - coverage)).clamp(-math.log(2), math.log(2))
        self.pending = None
        self.revealed_windows += 1
