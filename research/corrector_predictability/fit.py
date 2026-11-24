"""Matched sampling, development-only shrinkage, and checkpoint lineage."""
import copy
from pathlib import Path

import torch

from research.corrector.run import sha, table
from research.corrector_window_study.data import tensor_sha

from .alpha import moments, optimum, trajectory_indices
from .model import DIRECTION_ARMS, FEATURE_GROUPS, FeatureCorrector, PointCorrector, components

SEEDS = (42, 43, 44)
STEPS = (0, 50, 100, 250, 500, 750, 1000)
BATCH = 8


def network(mode, device):
    m = FeatureCorrector() if mode in FEATURE_GROUPS else PointCorrector()
    return m.to(device=device, memory_format=torch.channels_last)


def schedule(split, seed):
    trajectories = list(trajectory_indices(split.cases).values())
    rng = torch.Generator().manual_seed(seed)
    tracks = torch.randint(len(trajectories), (1000, BATCH), generator=rng).tolist()
    uniform = torch.rand(1000, BATCH, generator=rng).tolist()
    return torch.tensor([[trajectories[track][int(u*len(trajectories[track]))]
                          for track,u in zip(row, random)] for row,random in zip(tracks,uniform)])


@torch.inference_mode()
def correction(model, features, scale):
    model.eval()
    return torch.cat([model(*xs)*scale for xs in zip(*(p.split(BATCH) for p in features))])


def measured(pred, truth):
    c,e,native = moments(pred, truth)
    alpha = optimum(c.mean(), e.mean())
    return c,e,native, alpha


def candidate(row, indices, c, e, native, alpha, tag):
    mse = native + alpha**2*e - 2*alpha*c
    score = {name: float(mse[ix].mean()) for name,ix in indices.items()}
    return dict(row, alpha=alpha, alpha_mode=tag, development_mse=float(mse.mean()),
                development_improvement_pct=float(100*(1-mse.mean()/native.mean())),
                development_trajectory_mse=score)


class Loop:
    def __init__(self, train, development, train_inputs, dev_inputs, scale, out):
        self.train, self.dev = train, development
        self.tx, self.dx = train_inputs, dev_inputs
        self.scale = scale
        self.out = out
        self.log = []
        self.track = trajectory_indices(development.cases)
        self.truth = (train.y-train.p)/scale
        self.dev_truth = development.y.double()-development.p.double()
        self.train_truth = train.y.double()-train.p.double()

    def update(self, model, optim, plan, first, last, arm):
        model.train()
        running = []
        for i in plan[first:last]:
            optim.zero_grad(set_to_none=True)
            pred = model(*(x[i] for x in self.tx))
            a,b = components(pred, self.truth[i], arm)
            objective = a + .1*b
            objective.backward()
            norm = torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
            optim.step()
            running.append(torch.stack((a.detach(), b.detach(), norm.detach())))
        stats = torch.stack(running).mean(0)
        assert torch.isfinite(stats).all()
        return stats.tolist()

    def save(self, model, mode, arm, seed, step, plan_hash, stats):
        folder = self.out/"checkpoints"
        folder.mkdir(exist_ok=True)
        path = folder/f"{mode}_{arm}_{seed}_{step}.pt"
        assert not path.exists()
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, path)
        value = correction(model, self.dx, self.scale)
        c,e,native, alpha = measured(value, self.dev_truth)
        trained = correction(model, self.tx, self.scale)
        tc,te,tn = moments(trained, self.train_truth)
        row = dict(mode=mode, arm=arm, seed=seed, step=step,
                   checkpoint=str(path.relative_to(self.out)), sha256=sha(path),
                   schedule_sha256=plan_hash, training_mse=float((tn+te-2*tc).mean()),
                   train_native_mse=float(tn.mean()), dev_raw_mse=float((native+e-2*c).mean()),
                   dev_fitted_mse=float((native+alpha**2*e-2*alpha*c).mean()), dev_native_mse=float(native.mean()),
                   dev_alpha=alpha, dev_raw_inner=float(c.mean()), dev_raw_energy=float(e.mean()),
                   dev_raw_cosine=float((c/(e*native).sqrt().clamp_min(1e-30)).mean()),
                   train_loss_mse=stats[0], train_loss_direction=stats[1], train_gradient_norm=stats[2])
        self.log.append(row)
        print("CHECKPOINT", mode,arm,seed,step,f"raw={row['dev_raw_mse']:.10g}",
              f"cal={row['dev_fitted_mse']:.10g}",f"a={alpha:.5f}",flush=True)
        return [candidate(row,self.track,c,e,native,a,tag)
                for tag,a in (("fitted", alpha), ("raw", 1.), ("zero", 0.))]


def choose(all_rows):
    return min(all_rows, key=lambda r: (r["development_mse"], r["step"], r["alpha"]))


def choice_set(rows):
    return dict(selected=choose(rows),
                final_fit=next(r for r in rows if r["step"]==1000 and r["alpha_mode"]=="fitted"),
                final_raw=next(r for r in rows if r["step"]==1000 and r["alpha_mode"]=="raw"))


def standard(loop, mode, seed):
    torch.manual_seed(seed)
    model = network(mode, loop.train.x.device)
    stats = torch.optim.Adam(model.parameters(), lr=.001)
    sampling = schedule(loop.train, seed)
    digest = tensor_sha(sampling)
    sampling = sampling.to(loop.train.x.device)
    all_choices = loop.save(model, mode, "base", seed, 0, digest, (None, None, None))
    for a,b in zip(STEPS, STEPS[1:]):
        values = loop.update(model, stats, sampling, a, b, "base")
        all_choices.extend(loop.save(model,mode,"base",seed,b,digest,values))
    return choice_set(all_choices)


def directional(loop, base, seed):
    torch.manual_seed(seed)
    model = network(base, loop.train.x.device)
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    sampling = schedule(loop.train, seed)
    digest = tensor_sha(sampling)
    sampling = sampling.to(loop.train.x.device)
    candidates = []
    for first,last in zip((0, *STEPS[:3]), STEPS[:4]):
        values = (None,None,None) if last==0 else loop.update(model,optim,sampling,first,last,"base")
        candidates += loop.save(model,"Direction-"+base,"shared",seed,last,digest,values)
    weights, adam = copy.deepcopy(model.state_dict()),copy.deepcopy(optim.state_dict())
    result = {}
    for arm in DIRECTION_ARMS:
        model.load_state_dict(weights)
        optim.load_state_dict(copy.deepcopy(adam))
        assert {int(v["step"]) for v in optim.state.values()} == {250}
        options = list(candidates)
        for first,last in zip(STEPS[3:],STEPS[4:]):
            options += loop.save(model,"Direction-"+base,arm,seed,last,digest,
                                 loop.update(model,optim,sampling,first,last,arm))
        result["Direction-"+arm] = choice_set(options)
    return result


def read_model(out, mode, row, device):
    path = Path(out) / row["checkpoint"]
    assert sha(path)==row["sha256"]
    model = network(mode, device).eval().requires_grad_(False)
    if mode == "XP":
        # The SHA-pinned prior files also contain Adam and float metadata.
        payload = torch.load(path, map_location=device, weights_only=False)["model"]
    else:
        payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload)
    return model


def gate(configs, dev):
    tracks = trajectory_indices(dev.cases)
    base = {k: float((dev.y[ix].double()-dev.p[ix].double()).square().mean())
            for k,ix in tracks.items()}
    result = {}
    for mode,runs in configs.items():
        rows = [r["selected"] for r in runs]
        change = sum(r["development_improvement_pct"] for r in rows) / len(rows)
        positive = sum(r["development_improvement_pct"]>0 and r["alpha"]>0 and r["step"]>0
                       for r in rows)
        improved = sum(sum(r["development_trajectory_mse"][k] for r in rows)/len(rows)<b
                       for k,b in base.items())
        result[mode] = dict(improvement_pct=change, positive_nonzero_seeds=positive,
                            improved_trajectories=improved, passes=(change>=.5 and
                            positive>=2 and improved>=8))
    return result
