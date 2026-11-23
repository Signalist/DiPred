"""Development-only replay, same-weight route intervention and loss gradients."""

import gzip
import json
import time
from collections import Counter
from itertools import combinations

import numpy as np
import torch

from research.observed_residual_feedback.core import Publication, score_u
from research.observed_residual_feedback.reporting import avg, contrast, json_lines, trajectory_values
from research.observed_residual_feedback.run import (
    Observations,
    dump,
    noise_for,
    read,
    restore,
    score_rows,
    sha,
    tensor_hash,
    write_scores,
)
from research.residual_information_study.common.context import (
    CACHE, CACHE_BYTES, OUT, PREVIOUS, load_native, new_cache_bytes, remaining_gpu_seconds, schedule, windows,
)


FACTUAL = ("L", "J")
MASKED = "J_backbone_masked"
LOSS_NAMES = ("final_score_u", "weighted_backbone_score_u", "weighted_auxiliary")
GROUPS = ("up2_out", "encoder_projection", "head")
CHANNELS = (("overall", slice(None)), ("u", slice(0, 1)), ("v", slice(1, 2)))


def inner_product_diagnosis(residual, correction):
    """R and hatR here are already member means, with any scale applied."""
    residual_mse = float(residual.square().mean())
    correction_mse = float(correction.square().mean())
    final_mse = float((residual - correction).square().mean())
    inner = float((residual * correction).mean())
    identity_error = abs((final_mse - residual_mse) - (correction_mse - 2 * inner))
    assert identity_error <= 3e-13 * (1 + residual_mse + correction_mse)
    denom = (residual_mse * correction_mse) ** .5
    return dict(
        residual_rms=residual_mse ** .5,
        correction_rms=correction_mse ** .5,
        final_error_rms=final_mse ** .5,
        inner_product_per_element=inner,
        direction_cosine=inner / denom if denom else None,
        correction_mse=correction_mse,
        alignment_contribution=-2 * inner,
        mse_change=final_mse - residual_mse,
        identity_abs_error=identity_error,
    )


def decomposition_rows(backbone, correction, truth, std):
    residual = truth.double() - backbone.double().mean(dim=0)
    mean_correction = correction.double().mean(dim=0)
    for channel, sl in CHANNELS:
        for lead in (0, *range(1, 21)):
            span = slice(None) if lead == 0 else slice(lead - 1, lead)
            r = residual[sl, span]
            c = mean_correction[sl, span]
            for units, scale in (("standardized", 1.), ("physical", std[sl].double())):
                yield dict(channel=channel, lead=lead, units=units,
                           **inner_product_diagnosis(r * scale, c * scale))


def issue_record(stream, noise):
    return dict(
        history_sha256=tensor_hash(stream.history),
        noise_sha256=tensor_hash(noise),
        availability=stream.memory.available,
        previous_residual_sha256=tensor_hash(stream.memory.residual),
        previous_correction_sha256=tensor_hash(stream.memory.correction),
    )


def model_state_hashes(model):
    return {name: tensor_hash(tensor) for name, tensor in model.state_dict().items()}


@torch.no_grad()
def masked_backbone(model, stream, noise):
    """Use the existing L forward routing with J's exact parameters and facts."""
    assert model.arm == "J" and stream.memory.available and stream.pending is not None
    old_pending = stream.pending
    model.arm = "L"
    try:
        result = model(stream.history, noise, stream.memory)
    finally:
        model.arm = "J"
    assert stream.pending is old_pending
    return result


def source_records():
    records = {}
    with gzip.open(PREVIOUS / "publication.jsonl.gz", "rt") as handle:
        for line in handle:
            # Archived test events follow development in this append-only file.
            if '"split":"test"' in line:
                break
            row = json.loads(line)
            assert row["split"] == "development"
            if row["stage"] == 500 and row["arm"] in FACTUAL:
                records[(row["trajectory"], row["window"], row["arm"], row["event"])] = row
    assert len(records) == 144
    return records


def check_source_identity(current, archived, fields):
    for key in fields:
        assert current.get(key) == archived.get(key), (key, current, archived)


def issue_outputs(backbone, correction):
    return dict(backbone_sha256=tensor_hash(backbone), correction_sha256=tensor_hash(correction))


def score_outputs(records, identity, backbone, correction, truth, norm):
    for field, paths in (("backbone", backbone), ("final", backbone + correction)):
        records.extend(dict(**identity, field=field, **row) for row in score_rows(paths, truth, norm))


def verify_factual_scores(scores):
    """Check the numeric replay separately from exact issue tensor identities."""
    def key(row):
        return (row["trajectory"], row["window"], row["arm"], row["field"], row["channel"], row["lead"])

    archived = {key(row): row for row in json_lines(PREVIOUS / "development_500.jsonl.gz")
                if row["arm"] in FACTUAL}
    maximum = {}
    seen = set()
    largest = 0.0
    for row in scores:
        if row["arm"] not in FACTUAL:
            continue
        identity = key(row)
        assert identity not in seen
        seen.add(identity)
        source = archived[identity]
        for metric in ("score_u", "score_v", "crps_u", "crps_v", "mse", "smse", "variance"):
            if metric in row:
                delta = abs(row[metric] - source[metric])
                maximum[metric] = max(maximum.get(metric, 0.), delta)
                largest = max(largest, delta)
    assert seen == set(archived)
    assert largest <= 2e-5, ("Prior evaluation replay differs beyond tolerance", maximum)
    return dict(numeric_tolerance=2e-5, metric_rows=len(seen), max_abs_differences=maximum, max_abs_difference=largest)


@torch.no_grad()
def replay_facts(models, entries, norm, data_root, dest):
    archive = source_records()
    ledger, scores, decomposition = [], [], []
    counts = Counter()
    native_output_hashes = Counter()
    hashes_by_trajectory = []
    for entry in entries:
        if remaining_gpu_seconds() < 20:
            raise RuntimeError("Insufficient GPU budget for the next B trajectory")
        if new_cache_bytes() + 24 * 1024**2 > CACHE_BYTES:
            raise RuntimeError("Insufficient cache budget for the next B snapshot")
        obs = Observations(entry, norm, data_root)
        initial = obs.read_window(0)
        streams = {arm: Publication(initial) for arm in FACTUAL}
        snapshots = []
        prior_reveals = {}
        for n, seed in enumerate(entry["noise_seeds"], 1):
            common = dict(trajectory=entry["trajectory"], Re=entry["Re"], start=400, window=n, seed=seed)
            assert entry["start"] == 400
            noise = noise_for(initial, seed)
            inputs = {arm: issue_record(streams[arm], noise) for arm in FACTUAL}
            if n > 1:
                for arm in FACTUAL:
                    assert inputs[arm]["history_sha256"] == prior_reveals[arm]["truth_sha256"]
                    assert inputs[arm]["previous_residual_sha256"] == prior_reveals[arm]["residual_sha256"]
                    assert inputs[arm]["previous_correction_sha256"] == prior_reveals[arm]["mean_correction_sha256"]
            assert inputs["J"]["history_sha256"] == inputs["L"]["history_sha256"]
            snapshot = dict(**common, history=streams["J"].history.cpu(), noise=noise.cpu(), facts={})
            issued = {}
            current_hashes = {}
            for arm in FACTUAL:
                prior = streams[arm].memory
                snapshot["facts"][arm] = (prior.residual.cpu(), prior.correction.cpu(), prior.available)
                backbone, correction = streams[arm].issue(models[arm], noise)
                counts["factual"] += 1
                issued[arm] = (backbone, correction)
                hashes = dict(**inputs[arm], **issue_outputs(backbone, correction))
                current_hashes[arm] = hashes
                old = archive[(entry["trajectory"], n, arm, "issued")]
                check_source_identity(dict(**common, **hashes), old,
                                      ("trajectory", "window", "seed", "start", "history_sha256",
                                       "noise_sha256", "availability"))
                for field in ("previous_residual_sha256", "previous_correction_sha256",
                              "backbone_sha256", "correction_sha256"):
                    native_output_hashes[field] += hashes[field] == old[field]
                ledger.append(dict(**common, arm=arm, event="issued", utc_ns=time.time_ns(), **hashes,
                                   archived_tensor_hashes={field: old[field] for field in hashes
                                                           if field.endswith("_sha256")}))
            if n > 1:
                assert streams["J"].pending is not None
                assert issue_record(streams["J"], noise) == inputs["J"]
                issued[MASKED] = masked_backbone(models["J"], streams["J"], noise)
                assert issue_record(streams["J"], noise) == inputs["J"]
                counts["masked_warm"] += 1
                masked_hashes = issue_outputs(*issued[MASKED])
                ledger.append(dict(**common, arm=MASKED, event="issued", utc_ns=time.time_ns(),
                                   **inputs["J"], **masked_hashes, factual_arm="J",
                                   factual_backbone_sha256=current_hashes["J"]["backbone_sha256"],
                                   factual_correction_sha256=current_hashes["J"]["correction_sha256"]))
            snapshots.append(snapshot)
            truth = obs.read_window(n)
            truth_sha = tensor_hash(truth)
            reveal_ns = time.time_ns()
            assert reveal_ns > max(row["utc_ns"] for row in ledger[-len(issued):])
            for arm, (backbone, correction) in issued.items():
                identity = dict(**common, arm=arm)
                if arm in FACTUAL:
                    stream = streams[arm]
                    stream.reveal(truth)
                    memory = stream.memory
                    check_source_identity({"truth_sha256": truth_sha},
                                          archive[(entry["trajectory"], n, arm, "revealed")],
                                          ("truth_sha256",))
                    prior_reveals[arm] = dict(truth_sha256=truth_sha,
                                              residual_sha256=tensor_hash(memory.residual),
                                              mean_correction_sha256=tensor_hash(memory.correction))
                    ledger.append(dict(**identity, event="revealed", utc_ns=reveal_ns, **prior_reveals[arm]))
                else:
                    ledger.append(dict(**identity, event="scored_after_factual_reveal", utc_ns=reveal_ns,
                                       truth_sha256=truth_sha, state_updated=False))
                score_outputs(scores, identity, backbone, correction, truth, norm)
                if n > 1:
                    decomposition.extend(dict(**identity, **row) for row in
                                         decomposition_rows(backbone, correction, truth, norm["std"]))
                if arm == MASKED:
                    factual_backbone, factual_correction = issued["J"]
                    ledger.append(dict(
                        **identity, event="same_weight_differences",
                        backbone_member_rms=float((factual_backbone - backbone).double().square().mean().sqrt()),
                        correction_member_rms=float((factual_correction - correction).double().square().mean().sqrt()),
                        final_member_rms=float(((factual_backbone + factual_correction) -
                                               (backbone + correction)).double().square().mean().sqrt()),
                    ))
        path = dest / f"{entry['trajectory']}.pt"
        torch.save(snapshots, path)
        hashes_by_trajectory.append(dict(trajectory=entry["trajectory"], bytes=path.stat().st_size, sha256=sha(path)))
    assert counts == {"factual": 72, "masked_warm": 24}
    native_output_hashes = {field: dict(equal=count, total=72) for field, count in native_output_hashes.items()}
    return ledger, scores, decomposition, counts, native_output_hashes, hashes_by_trajectory


def fixed_gradient_entries():
    train = windows()["windows"]["train"]
    legal = [row for row in train if 400 in row["starts"]][:4]
    entries = read(OUT / "B_GRADIENT_SCHEDULE.json")
    assert len(legal) == len(entries) == 4
    for record, entry in zip(legal, entries):
        assert (entry["trajectory"], entry["Re"], entry["start"]) == (record["trajectory"], record["Re"], 400)
        assert len(entry["noise_seeds"]) == len(set(entry["noise_seeds"])) == 3
    return entries


def parameter_group(name):
    if name.startswith(("base.net.up2.", "base.net.out.")):
        return "up2_out"
    if name.startswith(("encoder.", "projection.")):
        return "encoder_projection"
    if name.startswith("head."):
        return "head"
    return None


def loss_gradient_rows(model, backbone, correction, truth, identity):
    numerical_aux = (correction.mean(0) - (truth - backbone.mean(0)).detach()).square().mean()
    ensemble_mse = (truth - (backbone + correction).mean(0)).square().mean()
    error = abs(float(numerical_aux.detach() - ensemble_mse.detach()))
    assert error < 2e-7
    losses = (
        score_u(backbone + correction, truth),
        .25 * score_u(backbone, truth),
        .1 * numerical_aux,
    )
    selected = [(name, parameter_group(name), parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    parameters = [parameter for _, _, parameter in selected]
    assert parameters and all(group in GROUPS for _, group, _ in selected)
    vectors = {}
    for i, (name, loss) in enumerate(zip(LOSS_NAMES, losses)):
        grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=i < len(losses) - 1)
        for group in GROUPS:
            vector = torch.cat([
                torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1)
                for (_, g, parameter), grad in zip(selected, grads) if g == group
            ]).double()
            vectors.setdefault(group, {})[name] = vector
    for group in GROUPS:
        grads = vectors[group]
        norms = {name: float(grad.norm()) for name, grad in grads.items()}
        cosines = {}
        for a, b in combinations(LOSS_NAMES, 2):
            denom = norms[a] * norms[b]
            cosines[f"{a} vs {b}"] = float(torch.dot(grads[a], grads[b])) / denom if denom else None
        yield dict(**identity, group=group,
                   weighted_loss_values={name: float(loss.detach()) for name, loss in zip(LOSS_NAMES, losses)},
                   auxiliary_unweighted_value=float(numerical_aux.detach()),
                   ensemble_mean_smse=float(ensemble_mse.detach()), identity_abs_error=error,
                   gradient_l2_norm=norms, gradient_cosine=cosines)


def inspect_gradients(model, entries, norm, data_root):
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    rows, forwards = [], 0
    for name, parameter in model.named_parameters():
        if parameter_group(name) is not None:
            parameter.requires_grad_(True)
    try:
        for entry in entries:
            if remaining_gpu_seconds() < 20:
                raise RuntimeError("Insufficient GPU budget for the next B gradient trajectory")
            obs = Observations(entry, norm, data_root)
            stream = Publication(obs.read_window(0))
            for n, seed in enumerate(entry["noise_seeds"], 1):
                with torch.set_grad_enabled(n > 1):
                    noise = noise_for(stream.history, seed)
                    assert not stream.history.requires_grad and not stream.memory.residual.requires_grad
                    assert not stream.memory.correction.requires_grad
                    backbone, correction = stream.issue(model, noise)
                    forwards += 1
                    truth = obs.read_window(n)
                    stream.reveal(truth)
                    if n > 1:
                        identity = dict(trajectory=entry["trajectory"], Re=entry["Re"], start=400, window=n, seed=seed)
                        rows.extend(loss_gradient_rows(model, backbone, correction, truth, identity))
                assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        model.requires_grad_(False)
    assert forwards == 12 and len(rows) == 24
    return rows, forwards


def paired_mse_decomposition(rows):
    """Warm window means first, with arithmetic decompositions stated explicitly."""
    groups = {}
    for row in rows:
        key = (row["arm"], row["channel"], row["lead"], row["units"])
        groups.setdefault(key, {}).setdefault(row["trajectory"], []).append(row)
    result = []
    for (arm, channel, lead, units), trajectories in groups.items():
        assert len(trajectories) == 12 and all(len(rows) == 2 for rows in trajectories.values())
        result_row = dict(arm=arm, channel=channel, lead=lead, units=units, trajectories=len(trajectories))
        for metric in ("residual_rms", "correction_rms", "final_error_rms", "inner_product_per_element",
                       "direction_cosine", "correction_mse", "alignment_contribution", "mse_change"):
            per_trajectory = {}
            for trajectory, observations in trajectories.items():
                values = [row[metric] for row in observations if row[metric] is not None]
                per_trajectory[trajectory] = float(np.mean(values)) if values else None
            defined = [x for x in per_trajectory.values() if x is not None]
            result_row[metric] = float(np.mean(defined)) if defined else None
            if metric == "mse_change":
                # Reuse the trajectory-level paired bootstrap with signed differences.
                paired = {trajectory: {"final": value, "backbone": 0.} for trajectory, value in per_trajectory.items()}
                result_row["mse_change_paired"] = contrast(paired, "final", "backbone")
        result_row["direction_cosine_defined_windows"] = sum(
            row["direction_cosine"] is not None for windows in trajectories.values() for row in windows)
        result.append(result_row)
    return result


def score_summary(scores):
    metrics = ("score_u", "score_v", "crps_u", "crps_v", "es_u", "es_v", "mse", "smse",
               "variance", "standardized_variance", "width90")
    summary = {}
    for arm in (*FACTUAL, MASKED):
        summary[arm] = {}
        for phase, windows in (("cold", (1,)), ("warm", (2, 3))):
            if arm == MASKED and phase == "cold":
                continue
            summary[arm][phase] = {}
            for metric in metrics:
                both = {}
                for field in ("backbone", "final"):
                    values = trajectory_values(scores, (arm,), windows, field, metric)
                    both[field] = values
                paired = {trajectory: {field: values[trajectory][arm] for field, values in both.items()}
                          for trajectory in both["final"]}
                summary[arm][phase][metric] = dict(
                    backbone=avg(both["backbone"], arm), final=avg(both["final"], arm),
                    final_minus_backbone=contrast(paired, "final", "backbone"))
    for field in ("backbone", "final"):
        summary[f"J_minus_masked_{field}"] = {}
        for metric in metrics:
            values = trajectory_values(scores, ("J", MASKED), (2, 3), field, metric)
            summary[f"J_minus_masked_{field}"][metric] = contrast(values, "J", MASKED)
    return summary


def gradient_summary(rows):
    result = {}
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        assert len(subset) == 8
        result[group] = {"gradient_l2_norm": {}, "gradient_cosine": {}}
        for metric in ("gradient_l2_norm", "gradient_cosine"):
            for name in subset[0][metric]:
                values = [row[metric][name] for row in subset if row[metric][name] is not None]
                result[group][metric][name] = dict(
                    mean=float(np.mean(values)) if values else None,
                    minimum=min(values) if values else None,
                    maximum=max(values) if values else None,
                    negative=int(sum(v < 0 for v in values)),
                    defined=len(values),
                )
    return result


def development():
    """Run B once. The study orchestrator owns the GPU lease and global budget."""
    dest = OUT / "correction_diagnosis"
    cache_dest = CACHE / "correction_diagnosis"
    if (dest / "summary.json").exists():
        return read(dest / "summary.json")
    dest.mkdir(parents=True, exist_ok=False)
    cache_dest.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    native, norm, frozen = load_native()
    old_run = read(PREVIOUS / "RUN.json")
    old_selection = read(PREVIOUS / "SELECTION.json")
    entries = schedule()["development"]
    assert len(entries) == 12 and old_run["common_updates"] == 500
    models, weight_hashes = {}, {}
    for arm in FACTUAL:
        assert old_selection["selected"][arm] == 500
        checkpoint = PREVIOUS / "weights" / f"{arm}_500.pt"
        claimed = [row["sha256"] for row in old_run["weights"][arm] if row["path"].endswith(f"{arm}_500.pt")]
        actual = sha(checkpoint)
        assert claimed == [actual]
        weight_hashes[arm] = actual
        models[arm] = restore(native, arm, 500)
        assert not models[arm].training and not any(p.requires_grad for p in models[arm].parameters())
    before = {arm: model_state_hashes(model) for arm, model in models.items()}
    ledger, scores, rows, forwards, matches, cache_rows = replay_facts(
        models, entries, norm, frozen["data_root"], cache_dest)
    gradient_entries = fixed_gradient_entries()
    gradients, forwards["gradient"] = inspect_gradients(models["J"], gradient_entries, norm, frozen["data_root"])
    torch.cuda.synchronize()
    assert new_cache_bytes() <= CACHE_BYTES
    assert sum(forwards.values()) == 108 and sum(forwards.values()) <= 120
    assert before == {arm: model_state_hashes(model) for arm, model in models.items()}
    assert all(not any(p.requires_grad or p.grad is not None for p in model.parameters()) for model in models.values())
    checks = verify_factual_scores(scores)
    decomposition = paired_mse_decomposition(rows)
    perturbation = {}
    for metric in ("backbone_member_rms", "correction_member_rms", "final_member_rms"):
        values = [row[metric] for row in ledger if row["event"] == "same_weight_differences"]
        assert len(values) == 24
        perturbation[metric] = dict(mean=float(np.mean(values)), minimum=min(values), maximum=max(values),
                                   nonzero=sum(x > 0 for x in values), windows=len(values))
    for name, records in (("publication", ledger), ("scores", scores), ("decomposition", rows),
                          ("decomposition_summary", decomposition), ("gradient", gradients)):
        write_scores(dest / f"{name}.jsonl.gz", records)
    summary = dict(scope="development_only", elapsed_seconds=time.monotonic() - start,
                   forward_groups=dict(**forwards, total=sum(forwards.values())), optimizer_updates=0,
                   checksum_verified_weights=weight_hashes, weights_unchanged=True,
                   cache_bytes=sum(row["bytes"] for row in cache_rows), factual_snapshots=cache_rows,
                   input_history_noise_truth_hashes_match_archive=True, factual_replay_tensor_matches=matches,
                   factual_replay_scores=checks, gradient_entries=gradient_entries,
                   gradient_stats=gradient_summary(gradients),
                   auxiliary_identity_max_abs=max(row["identity_abs_error"] for row in gradients),
                   decomposition_identity_max_abs=max(row["identity_abs_error"] for row in rows),
                   masked_perturbation=perturbation, scores=score_summary(scores),
                   warm_decomposition_overall=[row for row in decomposition
                                               if row["channel"] == "overall" and row["lead"] == 0])
    dump(dest / "summary.json", summary)
    return summary
