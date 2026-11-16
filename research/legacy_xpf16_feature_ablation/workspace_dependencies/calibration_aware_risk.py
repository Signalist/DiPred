"""Training objectives for raw or causally issued predictive-error intervals.

The local tail term is a training surrogate, not a coverage theorem. It changes
no inference outputs, parameters, or online controller rules.
"""

import torch


def pinball_field(lower, upper, residual):
    left, right = residual-lower, residual-upper
    return torch.maximum(.05*left,-.95*left)+torch.maximum(.95*right,-.05*right)


def unit_initial(prototype):
    return dict(shape=prototype.new_ones(1,2,1,1,1),
                amplitude=prototype.new_ones(1,2,1,1,1))


def group_scales(residual):
    """Fixed original-fit-label RMS scales, used by the loss only."""
    patches = residual.square().reshape(len(residual),2,20,4,8,8,8)
    patch = patches.mean((0,2,4,6),keepdim=False).sqrt().clamp_min(.01)[None]
    lead = residual.square().mean((0,3,4),keepdim=False).sqrt().clamp_min(.01)[None]
    return dict(patch=patch,lead=lead)


def local_tail_loss(field, scales):
    # Mean error score inside each fixed 8x8 region, over the full20lead block.
    patch = field.reshape(len(field),2,20,4,8,8,8).mean((2,4,6))
    patch = (patch/scales["patch"]).flatten(2)
    lead = field.mean((3,4))/scales["lead"]
    # A fixed upper quarter of spatial/lead groups. No threshold or weight scan.
    return .5*patch.topk(8,dim=-1).values.mean()+.5*lead.topk(5,dim=-1).values.mean()


def episode_indices(episodes, lead_window):
    """Map91trajectory segments and9successive windows into819cached fit pairs."""
    if not 0<=lead_window<9:
        raise ValueError("A fitting segment contains nine complete20frame predictions.")
    old = episodes<46
    return torch.where(old,lead_window*46+episodes,414+lead_window*45+(episodes-46))
