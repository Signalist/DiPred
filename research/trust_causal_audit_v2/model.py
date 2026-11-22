"""Equal-information routing; no training driver or optimizer.

G_context receives exactly the same current history, previous native mean,
local/neighbor risks, position, denoising label and native features as G_trust.
Only G_trust adds the fixed lambda=beta=1 routing prior.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .common import MEMBERS

NEIGHBORS = (0,1,2,3,5,6,7,8)


def neighbors(x):
    n,c,h,w = x.shape
    return F.unfold(x,3,padding=1).reshape(n,c,9,h,w)[:,:,NEIGHBORS]


def geometry(f, valid=None):
    n,_,h,w = f.shape
    valid = torch.ones(n,1,h,w,device=f.device,dtype=torch.bool) if valid is None else valid
    valid = valid.expand(n,1,h,w)
    assert valid.dtype == torch.bool
    return valid, neighbors(valid.to(f.dtype)).squeeze(1).bool() & valid


def routing(logits, q, risk, mask, trust):
    """Controlled monotonicity holds with learned logits/q held fixed.

    Learned heads also see risk, so this does not claim unconstrained full-model
    monotonicity with respect to a change in physical data or learned logits.
    """
    rj = neighbors(risk).squeeze(1)
    has_neighbor = mask.any(1,keepdim=True)
    adjusted = logits-rj if trust else logits
    safe = adjusted.masked_fill(~mask, -torch.inf)
    safe = torch.where(has_neighbor, safe, torch.zeros_like(safe))
    weights = torch.softmax(safe,1) * mask.to(logits.dtype)
    relative = risk - (weights*rj).sum(1,keepdim=True)
    gate = .25 * torch.sigmoid(q + relative if trust else q)
    gate = gate * has_neighbor.to(gate.dtype)
    return weights, gate


class SpatialRouter(nn.Module):
    def __init__(self, trust, seed=42):
        super().__init__()
        self.trust = bool(trust)
        # Same two pointwise heads as v1; expand their inputs to provide the
        # full observable context to BOTH arms.  No additional learned network.
        # f48 + H40 + prev_mean40 + xy2 + label1 + risk1 + risk_neighbors8 =140
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context = nn.Conv2d(140,8,1)
            self.gate = nn.Conv2d(140,1,1)
            nn.init.zeros_(self.context.bias)
            nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, f, history, previous_mean, label, risk, valid=None, enabled=True, trace=None):
        if not enabled:
            return f
        n,_,h,w = f.shape
        yy,xx = torch.meshgrid(torch.linspace(-1,1,h,device=f.device,dtype=f.dtype),
                              torch.linspace(-1,1,w,device=f.device,dtype=f.dtype), indexing='ij')
        coords = torch.stack((xx,yy))[None].expand(n,-1,-1,-1)
        step = (label.to(f.dtype)/99)[:,None,None,None].expand(n,1,h,w)
        _,mask = geometry(f,valid)
        # Native first features have spatial receptive fields. All 40 observed
        # history channels and the previous 40-channel mean additionally reach
        # the heads directly; this preserves physical-state compatibility cues.
        z = torch.cat((f,history.reshape(n,40,h,w),previous_mean.reshape(n,40,h,w),
                       coords,step,risk,neighbors(risk).squeeze(1)),1)
        logits,q = self.context(z),self.gate(z)
        weights,g = routing(logits,q,risk,mask,self.trust)
        correction = g * (weights[:,None]*(neighbors(f)-f[:,:,None])).sum(2)
        out = f+correction
        if trace is not None:
            if out.requires_grad: out.retain_grad()
            trace.append(dict(label=int(label[0]), output=out, weights=weights, gate=g,
                              logits=logits, q=q, mask=mask, learned_inputs=z))
        return out


class TrustDiffusion(nn.Module):
    def __init__(self, native, trust, seed=42):
        super().__init__()
        self.native = native.eval().requires_grad_(False)
        self.router = SpatialRouter(trust,seed).to(next(native.parameters()).device)

    def network(self,x,t,h,previous_mean,risk,valid=None,enabled=True,trace=None):
        net = self.native.net
        b,_,_,ny,nx = h.shape
        assert h.shape == x.shape and h.shape[1:] == (2,20,32,64)
        increments = torch.cat((torch.zeros_like(h[:,:,:1]),h.diff(dim=2)),2)/net.scale
        u = t.to(h.dtype)/99
        embedding = net.time(torch.stack((u,u*u,torch.sin(torch.pi*u),torch.cos(torch.pi*u)),1))
        inputs = torch.cat((h.reshape(b,40,ny,nx),x.reshape(b,40,ny,nx),increments.reshape(b,40,ny,nx)),1)
        first = net.first(inputs,embedding)
        first = self.router(first,h,previous_mean,t,risk,valid,enabled,trace)
        encoded = net.enc1(net.down1(first),embedding)
        middle = net.mid(net.down2(encoded),embedding)
        upper = net.up1(torch.cat((F.interpolate(middle,size=encoded.shape[-2:],mode='bilinear',align_corners=False),encoded),1),embedding)
        upper = net.up2(torch.cat((F.interpolate(upper,size=first.shape[-2:],mode='bilinear',align_corners=False),first),1),embedding)
        return net.out(upper).reshape_as(h)

    def sample(self,history,previous_mean,risk,noise,enabled=True,trace=None,valid=None):
        assert len(history)==len(noise) and previous_mean.shape==history.shape
        x = noise.clone()
        for step in (3,2,1,0):
            label = torch.full((len(history),),step*33,device=x.device,dtype=torch.long)
            # Even with fusion OFF, exercise the wrapper's entire native path.
            clean = self.network(x,label,history,previous_mean,risk,valid,enabled,trace)
            alpha = self.native.abar[step]
            eps = (x-alpha.sqrt()*clean)/(1-alpha).sqrt()
            prev = self.native.abar[step-1] if step else x.new_tensor(1.)
            x = prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x


def native_sample(native,history,noise):
    x = noise.clone()
    for step in (3,2,1,0):
        label = torch.full((len(history),),step*33,device=x.device,dtype=torch.long)
        clean = native.net(x,label,history)
        alpha = native.abar[step]
        eps = (x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        prev = native.abar[step-1] if step else x.new_tensor(1.)
        x = prev.sqrt()*clean+(1-prev).sqrt()*eps
    return x


def crps_parts(paths,truth):
    n = len(paths)
    weights = (2*torch.arange(n,device=paths.device,dtype=paths.dtype)-n+1).reshape(n,*([1]*(paths.ndim-1)))
    first = (paths-truth).abs().mean()
    pair_half = (paths.sort(0).values*weights).sum(0).mean()/(n*n)
    return first-pair_half,first,pair_half


def energy_score(paths,truth,std=None):
    """Full-path RMS Energy Score; std=None is physical units, else normalized."""
    if std is not None:
        assert std.shape==(2,1,1,1)
        paths,truth = paths/std,truth/std
    a = (paths-truth).flatten(1).square().mean(1).sqrt().mean()
    pair = (paths[:,None]-paths[None,:]).flatten(2).square().mean(2).sqrt()
    return a-.5*pair.mean()
