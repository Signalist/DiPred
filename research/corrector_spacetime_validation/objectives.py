"""Auxiliary objectives and the exact finite-window temporal operator."""
import torch

from research.corrector_spacetime_optimization.loss import extensions

T = 20
RHO = (.1,.3)
KINDS = dict(TS1="velocity",TS2="acceleration",TM1="match",TM2="match2")
ORDER = tuple(torch.randperm(T,generator=torch.Generator().manual_seed(42)).tolist())


def operator(dtype=torch.float64):
    return torch.eye(T,dtype=dtype).diff(n=2,dim=0)


def terms(r,y,names):
    original = tuple(k for k in names if k not in ("energy","diag","perm"))
    values = extensions(r,y,original)
    if "energy" in names:
        values["energy"] = r.square().mean()
    if "diag" in names:
        h = operator().square().sum(0).to(r).view(1,1,T,1,1)
        values["diag"] = ((r-y).square()*h).mean()*T/(T-2)
    if "perm" in names:
        values["perm"] = (r-y)[:,:,ORDER].diff(n=2,dim=2).square().mean()
    return values


def objective(r,y,weights):
    values = terms(r,y,weights)
    return values["mse"]+sum(values[k]*v for k,v in weights.items())


def verify_math():
    rng = torch.Generator().manual_seed(42)
    r,y = (torch.randn(3,2,T,5,6,generator=rng,dtype=torch.float64) for _ in range(2))
    e = r-y
    D = operator()
    H = D.T@D
    v = terms(r,y,("match","match2","acceleration","energy","diag","perm"))
    sequences = e.movedim(2,-1).flatten(0,-2)
    lam = .73
    matrix = torch.eye(T,dtype=e.dtype)/T + lam*H/(T-2)
    errors = dict(tm1=float(abs(v["match"]-e.diff(dim=2).square().mean())),
                  tm2=float(abs(v["match2"]-e.diff(n=2,dim=2).square().mean())),
                  quadratic=float(abs(v["mse"]+lam*v["match2"]-(sequences@matrix*sequences).sum(1).mean())),
                  energy_target=float(abs(v["mse"]+lam*v["energy"]-
                                    ((1+lam)*(r-y/(1+lam)).square().mean()+lam/(1+lam)*y.square().mean()))))
    assert max(errors.values())<1e-12 and torch.linalg.eigvalsh(matrix).min()>0
    assert float(v["match2"])!=float(v["acceleration"])
    assert (H-torch.diag(H.diag())).abs().sum()>0
    return dict(errors=errors,diagonal=H.diag().tolist(),permutation=list(ORDER),eigenvalues=torch.linalg.eigvalsh(H).tolist())
