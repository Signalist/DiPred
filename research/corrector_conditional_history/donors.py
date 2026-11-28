"""Frozen, condition- and time-matched donor indices, without target access."""
from collections import Counter, defaultdict

import numpy as np
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import write_tensor
from research.corrector_window_study.data import tensor_sha


def candidates(cases, slack):
    same_regime = defaultdict(list)
    for i, case in enumerate(cases):
        same_regime[case["re"]].append(i)
    options = []
    for i, case in enumerate(cases):
        choices = [(abs(case["start"]-cases[j]["start"]), j) for j in same_regime[case["re"]]
                   if cases[j]["file"] != case["file"]]
        if choices:
            closest = min(distance for distance, _ in choices)
            options.append([j for distance, j in choices if distance <= closest+slack])
        else:
            options.append([i])
    return options


def assign(cases, recipients, seed, slack):
    pool = candidates(cases, slack)
    random = np.random.default_rng(seed)
    indices = [p[random.integers(len(p))] for p in pool]
    result = np.empty(recipients.shape, dtype=np.int64)
    for i, j in enumerate(recipients.ravel().tolist()):
        nearest = pool[j]
        result.flat[i] = indices[j] if recipients.ndim == 1 else nearest[random.integers(len(nearest))]
    for i, j in zip(recipients.ravel(), result.ravel()):
        assert cases[i]["re"] == cases[j]["re"]
        assert i == j or cases[i]["file"] != cases[j]["file"]
    return torch.from_numpy(result)


def prepare(out, scope, view, schedule, seed, slack):
    recipients = schedule.cpu().numpy()
    cases = view.data.cases
    chosen = assign(cases, recipients, seed, slack)
    path = out / f"donor_batches_{scope}.pt"
    if path.exists():
        assert torch.equal(torch.load(path, weights_only=True), chosen)
    else:
        write_tensor(path, chosen)
    usage = Counter(chosen.flatten().tolist())
    paired = Counter(zip(recipients.ravel().tolist(), chosen.flatten().tolist()))
    rows = [dict(scope=scope, recipient=i, donor=j, count=n, re=cases[i]["re"],
                 recipient_file=cases[i]["file"], donor_file=cases[j]["file"],
                 recipient_start=cases[i]["start"], donor_start=cases[j]["start"], moved=i!=j)
            for (i, j), n in sorted(paired.items())]
    history = view.true
    remaining = history - view.historical.double()
    summaries = {}
    for name, field in (("R", history), ("E", remaining)):
        energy = field.square().mean((1,2,3,4)).cpu().numpy()
        a, b = energy[recipients], energy[chosen.numpy()]
        summaries[name] = dict(actual_mean_energy=float(a.mean()), donor_mean_energy=float(b.mean()),
                               actual_quantiles=np.quantile(a,[.1,.5,.9]).tolist(),
                               donor_quantiles=np.quantile(b,[.1,.5,.9]).tolist())
    total = int(chosen.numel())
    return chosen, rows, dict(file=path.name, sha256=sha(path), tensor_sha256=tensor_sha(chosen),
                              sampling_sha256=tensor_sha(schedule), assignments=total,
                              moved=sum(r["count"] for r in rows if r["moved"]), same_re=total,
                              max_donor_uses=max(usage.values()), unique_donors=len(usage),
                              mean_abs_start_gap=sum(abs(r["recipient_start"]-r["donor_start"])*r["count"]
                                                     for r in rows)/total, distributions=summaries)
