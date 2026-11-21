"""Trajectory-paired summaries and independent checks of the frozen followup."""
from collections import defaultdict
import json

from run import (ROOT, OUT, OLD, CONFIG, PROTOCOL_HASH, METRICS, RECORDS, digest,
                 read_csv, csvwrite, dump, compare_rows, np)
from common import bootstrap


def numeric(rows):
    keys = set(METRICS) | {"inherited", "accumulated", "cross", "total", "raw", "train_centered"}
    return [{key: float(value) if key in keys else value for key, value in row.items()} for row in rows]


def average(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append([float(row[metric]) for metric in metrics])
    return [dict(zip(keys, key), **dict(zip(metrics, np.mean(values, axis=0))))
            for key, values in groups.items()]


def paired(rows, scopes, metrics):
    result = []
    for scope in scopes:
        # Windows and fixed-head scores are averaged inside each original trajectory.
        reduced = average(rows, [*scope, "trajectory", "method"], metrics)
        groups = defaultdict(dict)
        for row in reduced:
            groups[tuple(row[key] for key in scope)][row["trajectory"], row["method"]] = row
        for values, index in groups.items():
            trajectories = sorted({trajectory for trajectory, method in index if method == "crps"})
            for metric in metrics:
                differences = [index[trajectory, "crps"][metric] - index[trajectory, "native"][metric]
                               for trajectory in trajectories]
                baseline = float(np.mean([index[trajectory, "native"][metric] for trajectory in trajectories]))
                estimate = bootstrap(differences)
                result.append(dict(scope="overall" if not scope else "+".join(scope),
                                   subgroup="all" if not scope else "|".join(values), metric=metric,
                                   native=baseline, corrected=baseline + estimate["mean_delta"],
                                   relative_percent=100 * estimate["mean_delta"] / baseline if baseline else "",
                                   mean_delta=estimate["mean_delta"], ci95_low=estimate["ci95"][0],
                                   ci95_high=estimate["ci95"][1], improved=estimate["improved"],
                                   trajectories=estimate["trajectories"]))
    return result


def main():
    scores = numeric(read_csv(OUT / "scores.csv"))
    detail = numeric(read_csv(OUT / "lead_channel.csv"))
    assert len(scores) == 12 * 20 * 4 and len(detail) == 40 * len(scores)
    identities = ["trajectory", "start", "method", "seed"]
    assert len({tuple(row[key] for key in identities) for row in scores}) == len(scores)
    expected = {(record["file"], str(start), method, str(seed))
                for record in RECORDS["test"] for start in CONFIG["starts"]
                for method, seed in [("native", 0), *(("crps", seed) for seed in CONFIG["head_seeds"])]}
    assert {tuple(row[key] for key in identities) for row in scores} == expected
    for row in scores:
        assert all(np.isfinite(row[metric]) for metric in METRICS)
    cells = average(detail, identities, ["crps", "mse"])
    detail_check = compare_rows(cells, scores, identities, ["crps", "mse"], "detail-to-block")
    for keys, name in [
        (["method"], "summary"), (["method", "seed"], "per_seed"),
        (["method", "trajectory"], "per_trajectory"), (["method", "trajectory", "seed"], "per_trajectory_seed"),
        (["method", "Re"], "per_Re"), (["method", "start"], "per_window"),
        (["method", "Re", "seed"], "per_Re_seed"),
    ]:
        csvwrite(OUT / f"{name}.csv", average(scores, keys, METRICS))
    for keys, name in [
        (["method", "lead"], "per_lead"), (["method", "channel"], "per_channel"),
        (["method", "Re", "channel"], "per_Re_channel"),
        (["method", "lead", "channel"], "per_lead_channel"),
        (["method", "Re", "lead", "channel"], "per_Re_lead_channel"),
    ]:
        csvwrite(OUT / f"{name}.csv", average(detail, keys, ["crps", "mse"]))
    comparisons = paired(scores, [[], ["Re"], ["start"]], METRICS)
    comparisons += paired(detail, [["lead"], ["channel"], ["Re", "channel"]], ["crps", "mse"])
    for seed in CONFIG["head_seeds"]:
        selected = [row for row in scores if row["method"] == "native" or row["seed"] == str(seed)]
        for row in paired(selected, [[]], METRICS):
            row["scope"], row["subgroup"] = "seed", str(seed)
            comparisons.append(row)
    csvwrite(OUT / "paired.csv", comparisons)
    trajectory_means = average(scores, ["trajectory", "Re", "method"], METRICS)
    index = {(row["trajectory"], row["method"]): row for row in trajectory_means}
    differences = []
    for record in RECORDS["test"]:
        native = index[record["file"], "native"]
        corrected = index[record["file"], "crps"]
        differences.append(dict(trajectory=record["file"], Re=record["group"],
                                **{metric: corrected[metric] - native[metric] for metric in METRICS}))
    csvwrite(OUT / "trajectory_differences.csv", differences)
    native = [row for row in scores if row["method"] == "native"]
    observations = {(row["trajectory"], row["start"]): row["event"] for row in native}
    for row in scores:
        assert row["event"] == observations[row["trajectory"], row["start"]]
        assert abs(row["brier"] - (float(row["probability"]) - int(row["event"])) ** 2) <= 1e-15
    positives = [row for row in native if int(row["event"])]
    dump(OUT / "events_summary.json", dict(blocks=len(native), trajectories=len(RECORDS["test"]),
         positive_blocks=len(positives), positive_trajectories=len({row["trajectory"] for row in positives}),
         threshold=CONFIG["event_threshold"], positive_cases=positives))

    # Existing stream provides an independent native score reference for every block.
    old_stream = read_csv(ROOT / "results/residual_dynamics_stream/per_window.csv")
    stream_native = [dict(row, trajectory=row["file"]) for row in old_stream if row["condition"] == "native"]
    stream_check = compare_rows(native, stream_native, ["trajectory", "start"], METRICS, "old stream native scores")

    decompositions = numeric(read_csv(OUT / "decomposition.csv"))
    terms = ["inherited", "accumulated", "cross", "total"]
    assert len(decompositions) == (24 + 24 + 240) * 40
    for row in decompositions:
        assert abs(row["total"] - row["inherited"] - row["accumulated"] - row["cross"]) <= CONFIG["tolerances"]["decomposition_energy"]
    for keys, name in [
        (["split", "lead"], "decomposition_by_lead"),
        (["split", "Re", "lead", "channel"], "decomposition_by_Re_lead_channel"),
        (["split", "Re", "channel"], "decomposition_by_Re_channel"),
        (["split", "trajectory"], "decomposition_by_trajectory"),
        (["split", "channel"], "decomposition_by_channel"),
        (["split"], "decomposition_summary"),
    ]:
        csvwrite(OUT / f"{name}.csv", average(decompositions, keys, terms))
    correlations = numeric(read_csv(OUT / "innovation_correlations.csv"))
    assert len(correlations) == (24 + 24 + 240) * 2 * 19
    assert all(np.isfinite(row[key]) for row in correlations for key in ["raw", "train_centered"])
    csvwrite(OUT / "correlations_by_Re_lead_channel.csv",
             average(correlations, ["split", "Re", "lead", "channel"], ["raw", "train_centered"]))
    pooled = [row for row in correlations if row["lead"] == "0"]
    csvwrite(OUT / "correlations_summary.csv", average(pooled, ["split", "channel"], ["raw", "train_centered"]))

    invariants = read_csv(OUT / "shift_invariance.csv")
    assert len(invariants) == 20 * 12 * 3
    maximum = {key: max(float(row[key]) for row in invariants)
               for key in invariants[0] if key not in ("trajectory", "start", "seed")}
    for key, value in maximum.items():
        tolerance = CONFIG["tolerances"]["covariance" if "covariance" in key or "gram" in key else "shift_fields"]
        assert value <= tolerance
    manifest = json.loads((OUT / "cache_manifest.json").read_text())
    assert [row["identity"]["start"] for row in manifest] == CONFIG["starts"]
    for row in manifest:
        start = row["identity"]["start"]
        assert digest(OUT / "cache" / f"issued_{start}.pt") == row["sha256"]
        assert row["identity"]["protocol_sha256"] == PROTOCOL_HASH
    log = [json.loads(line) for line in (OUT / "events.jsonl").read_text().splitlines()]
    for start in CONFIG["starts"]:
        entries = [row["action"] for row in log if row["start"] == start]
        pending = False
        for action in entries:
            if action in ("issued_all32_and_all3_heads_before_truth", "reuse_issued_without_current_truth"):
                assert not pending
                pending = True
            elif action == "revealed_complete20":
                assert pending
                pending = False
        assert not pending
        assert entries
    assert json.loads((OUT / "reproduction.json").read_text())["passed"]
    dump(OUT / "verification.json", dict(passed=True, scores=len(scores), detail=len(detail),
        original_trajectories=12, windows=20, invariance_maximum=maximum,
        detail_check=detail_check, independent_stream_check=stream_check,
        protocol_sha256=PROTOCOL_HASH, cache_hashes_verified=20,
        prediction_before_truth_verified=True))
    print("PASS 20 windows, 12 original trajectories, 3 frozen heads, full detail and paired summaries", flush=True)
    print("PASS prior stream native reference, event counts, covariance, hashes and issue/reveal sequence", flush=True)
    print(json.dumps([row for row in comparisons if row["scope"] == "overall"], indent=2))
    print("event", len(positives), "blocks", len({row["trajectory"] for row in positives}), "trajectories")


if __name__ == "__main__":
    main()
