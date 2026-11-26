"""Paired correction heads with fixed output subspaces and optional history maps."""
import torch
from torch import nn

from research.corrector_residual_feedback.model import model
from research.corrector_spacetime_validation.objectives import operator,objective


def spec(family,history,weights=None,rank=20,mix="original",sequence=None):
    return dict(family=family,history=history,weights=weights or {},rank=rank,mix=mix,sequence=sequence)


def temporal_basis():
    d=operator()
    h=d.T@d
    _,v=torch.linalg.eigh(h)
    linear=torch.arange(20,dtype=torch.float64)-9.5
    basis=v.clone()
    basis[:,0]=1/(20**.5)
    basis[:,1]=linear/linear.norm()
    for j in range(2,20):
        if basis[basis[:,j].abs().argmax(),j]<0:
            basis[:,j].neg_()
    assert torch.allclose(basis.T@basis,torch.eye(20,dtype=basis.dtype),atol=1e-12,rtol=0)
    return basis


def verify_basis(basis):
    t=torch.randn((2,20),generator=torch.Generator().manual_seed(42),dtype=basis.dtype)
    identity=basis@basis.T
    assert torch.allclose(t@identity,t,atol=1e-12,rtol=0)
    assert torch.allclose((t@basis).square().mean(),t.square().mean(),atol=1e-12,rtol=0)
    return dict(identity_error=float((t@identity-t).abs().max()),
                mse_error=float(abs((t@basis).square().mean()-t.square().mean())))


class Head(nn.Module):
    def __init__(self,cfg,basis,device):
        super().__init__()
        self.cfg=cfg
        self.cnn=model(8+2*cfg["history"],device)
        nn.init.zeros_(self.cnn[-1].weight)
        nn.init.zeros_(self.cnn[-1].bias)
        mix=cfg["mix"]
        i,j=torch.meshgrid(torch.arange(1,21,dtype=torch.float32,device=device),
                           torch.arange(1,21,dtype=torch.float32,device=device),indexing="ij")
        lag=(20+i-j)/39
        self.register_buffer("coordinates",lag[...,None] if mix=="lag" else torch.stack((j/20,i/20,lag),-1),
                             persistent=False)
        self.register_buffer("eye",torch.eye(20,device=device),persistent=False)
        if mix in ("lag","dual"):
            self.mlp=nn.Sequential(nn.Linear(1 if mix=="lag" else 3,8),nn.SiLU(),nn.Linear(8,1)).to(device)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        if mix=="free":
            self.delta=nn.Parameter(torch.zeros(20,20,device=device))
        rank=cfg["rank"]
        project=basis[:,:rank]@basis[:,:rank].T
        self.register_buffer("projector",project.to(device=device,dtype=torch.float32),persistent=False)

    def mapping(self):
        kind=self.cfg["mix"]
        if kind=="last":
            return self.eye[-1:].expand(20,-1)
        if kind=="reverse":
            return self.eye.flip(1)
        if kind in ("lag","dual"):
            return self.eye+self.mlp(self.coordinates).squeeze(-1)
        if kind=="free":
            return self.eye+self.delta
        assert kind=="original"
        return self.eye

    def forward(self,z):
        if self.cfg["mix"]!="original":
            mixed=torch.einsum("ji,bcixy->bcjxy",self.mapping(),z[:,8:10])
            z=torch.cat((z[:,:8],mixed,z[:,10:]),dim=1)
        r=self.cnn(z.contiguous())
        if self.cfg["rank"]<20:
            r=torch.einsum("ji,bcixy->bcjxy",self.projector,r)
        return r


def inputs(view,ix,cfg,feedback=None,intervened=None):
    fields=[view.base[ix],(view.current[ix]/view.scale).float()]
    if cfg["history"]>=1:
        raw=view.true[ix] if intervened is None else intervened
        fields.append((raw/view.scale).float())
    if cfg["history"]==2:
        previous=view.historical[ix] if feedback is None else feedback
        fields.append((previous/view.scale).float())
    return torch.cat(fields,dim=1)


def labels(view,ix,cfg):
    p=view.target[ix]
    if cfg["family"]=="I":
        p=p-view.current[ix].double()
    return (p/view.scale).float()


def residual(view,ix,cfg,correction):
    out=correction.double()*view.scale
    return out+view.current[ix].double() if cfg["family"]=="I" else out


def train_loss(net,view,ix,cfg):
    if ix.ndim==1:
        return objective(net(inputs(view,ix,cfg)),labels(view,ix,cfg),cfg["weights"])
    losses=[]
    previous=None
    for k in range(ix.shape[1]):
        windows=ix[:,k]
        estimate=net(inputs(view,windows,cfg,feedback=previous))
        losses.append(objective(estimate,labels(view,windows,cfg),cfg["weights"]))
        if cfg["sequence"]=="self":
            previous=residual(view,windows,cfg,estimate).detach()
    return torch.stack(losses).mean()
