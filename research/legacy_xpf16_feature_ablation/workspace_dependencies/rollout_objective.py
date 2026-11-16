"""Forecast-risk training; stochastic refinement, not score matching."""
import torch
import torch.nn.functional as F

def forecast(model,h,refine=False,generator=None):
    if h.shape[2]!=20:raise ValueError('Exactly twenty observed frames required')
    history=h;blocks=[]
    for _ in range(4):
        block=model.next_block(history,refine,generator)
        blocks.append(block);history=torch.cat([history[:,:,5:],block],2)
    return torch.cat(blocks,2)

def loss(model,h,y,refine=False):
    if y.shape[2]!=20:raise ValueError('Exactly twenty future targets required')
    return F.mse_loss(forecast(model,h,refine),y)
