"""Frozen four-call known-field features; no future labels."""

import torch
import torch.nn.functional as F


BASE_CHANNELS = 14


def grid_derivative(x, axis, spacing):
    shape = x.shape
    if axis == -1:
        derivative = x.diff(dim=-1).reshape(-1, 1, shape[-2], shape[-1] - 1)
        derivative = F.pad(derivative, (0, 1, 0, 0), mode="replicate").reshape(shape)
    else:
        derivative = x.diff(dim=-2).reshape(-1, 1, shape[-2] - 1, shape[-1])
        derivative = F.pad(derivative, (0, 0, 0, 1), mode="replicate").reshape(shape)
    return derivative / spacing.reshape(-1, 1, 1, 1, 1)


def base_features(history, prediction, mean, std, geometry):
    h = (history - mean) / std
    p = (prediction - mean) / std
    last = h[:, :, -1:].expand_as(p)
    variability = h.std(dim=2, keepdim=True).expand_as(p)
    departure = p - last
    dx = grid_derivative(p, -1, torch.ones_like(geometry[:, 0]))
    dy = grid_derivative(p, -2, torch.ones_like(geometry[:, 1]))
    gradient = (dx.square() + dy.square() + 1e-8).sqrt()
    change = torch.cat([p[:, :, :1] - h[:, :, -1:], p.diff(dim=2)], dim=2).abs()
    # Coordinate-grid curl proxy; no pressure/viscosity residual is claimed.
    raw_dx = grid_derivative(prediction, -1, geometry[:, 0])
    raw_dy = grid_derivative(prediction, -2, geometry[:, 1])
    curl = raw_dx[:, 1:2] - raw_dy[:, :1]
    lead = torch.arange(1, 21, device=p.device, dtype=p.dtype).reshape(1, 1, 20, 1, 1) / 20
    lead = lead.expand(len(p), 1, 20, *p.shape[-2:])
    features = torch.cat([p, last, variability, departure, gradient, change, curl, lead], dim=1)
    assert features.shape[1] == BASE_CHANNELS
    return features


def risk_features(history,prediction,raw_width):
    """Inputs use normalized physical fields, all available at forecast issue."""
    b,c,t,h,w=prediction.shape
    dx=F.pad(prediction.diff(dim=-1),(0,1,0,0))
    dy=F.pad(prediction.diff(dim=-2),(0,0,0,1))
    gradient=(dx.square()+dy.square()+1e-8).sqrt()
    departure=(prediction-history[:,:,-1:]).abs()
    variability=history.std(dim=2,keepdim=True).expand_as(prediction)
    change=torch.cat([prediction[:,:,:1]-history[:,:,-1:],prediction.diff(dim=2)],2).abs()
    lead=torch.arange(1,t+1,device=prediction.device,dtype=prediction.dtype)[None,None,:,None,None]/t
    lead=lead.expand_as(prediction)
    yy=torch.linspace(-1,1,h,device=prediction.device,dtype=prediction.dtype)
    xx=torch.linspace(-1,1,w,device=prediction.device,dtype=prediction.dtype)
    y,x=torch.meshgrid(yy,xx,indexing="ij")
    x=x[None,None,None].expand_as(prediction)
    y=y[None,None,None].expand_as(prediction)
    values=[value.clamp_min(1e-4).log() for value in (raw_width,gradient,departure,variability,change)]
    values.extend([prediction.abs(),lead,lead.square(),x,y,x*y])
    value=torch.stack(values,dim=2)
    mean=value.mean((3,4,5),keepdim=True)
    std=value.std((3,4,5),keepdim=True).clamp_min(.05)
    value=((value-mean)/std).clamp(-5,5)
    return value-value.mean((3,4,5),keepdim=True)


def adapter_inputs(history, point, lower, upper, previous_point):
    width = (0.5 * (upper - lower)).clamp_min(1e-6)
    known = risk_features(history, point, width)[:, :, :5]
    known = known.reshape(len(point), 10, 20, 32, 64)
    available = previous_point is not None
    if available:
        # The current history is the revealed target of the previous forecast.
        previous_error = (history - previous_point).abs()
        remembered = ((previous_error + 1e-4) / width).log().clamp(-5, 5)
    else:
        remembered = torch.zeros_like(point)
    lead = torch.linspace(0.05, 1, 20, device=point.device, dtype=point.dtype)[None, None, :, None, None]
    lead = lead.expand(len(point), 1, 20, 32, 64)
    mask = torch.full_like(lead, float(available))
    return torch.cat([known, remembered, lead, mask], 1)


def cue_inputs(history,point,lower,upper,change,cue,available):
    value=adapter_inputs(history,point,lower,upper,None)
    value[:,13:14]=float(available)
    width=(0.5*(upper-lower)).clamp_min(1e-6)
    if cue=="history_change":
        signal=(history[:,:,-1:]-history[:,:,-2:-1]).abs().expand_as(point)
    elif cue=="denoising_change":
        signal=change
    elif cue=="none":
        signal=None
    else:
        raise ValueError("Choose none,history_change,denoising_change.")
    if signal is not None:
        value[:,10:12]=torch.log1p(signal/width).clamp(0,5)
    return value


def forecast_features(model, normalized_history):
    stages = []
    hook = model.net.up2.register_forward_hook(
        lambda module, inputs, output: stages.append(output.detach().clone()))
    try:
        prediction = model.rollout(normalized_history, 500000, sign=0)
    finally:
        hook.remove()
    assert len(stages) == 4
    assert all(stage.shape[1:] == (48, 32, 64) for stage in stages)
    # Call order corresponds to diffusion indices3,2,1,0; PDE future stays20.
    return prediction, torch.cat(stages, dim=1)
