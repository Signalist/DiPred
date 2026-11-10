"""Independently recompute all366 original96-cell comparisons and failures.

Only the standard library is required. Historic reference summaries remain
frozen; all new warm-group metrics are regenerated from complete window rows.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / "runs/dynamic_risk_memory_followup_20261110"
METRICS = {
    "standardized_interval_score": -1, "standardized_width": -1,
    "top10_energy": 1, "per_frame_top10_energy": 1, "relative_aurc": -1,
    "worst_frame_coverage": 1, "worst_patch_coverage": 1, "coverage_error": -1,
}
NAMES = ("frozen_best", "modulate", "warp")
MODES = ("memory_dynamic", "scale_fastcal")
TOLERANCE = 1e-6


def key(row):
    return tuple(row[name] for name in ("split", "group", "channel", "map", "variant"))


def primary_value(row, metric):
    return abs(row["coverage"] - .9) if metric == "coverage_error" else row[metric]


def main():
    cases = json.loads((OUT / "per_case.json").read_text())
    assert len(cases) == 11232
    unique, points = set(), {}
    buckets = defaultdict(list)
    for row in cases:
        identity = tuple(row[name] for name in ("split", "file", "start", "channel", "map", "variant"))
        assert identity not in unique
        unique.add(identity)
        assert row["mse"] == points.setdefault(identity[:4], row["mse"])
        if row["start"] >= 20:
            buckets[key(row)].append(row)
    numeric = [name for name, value in cases[0].items() if isinstance(value, float)]
    computed = {}
    for identity, rows in buckets.items():
        assert set(Counter(row["file"] for row in rows).values()) == {38}
        computed[identity] = {metric: mean(row[metric] for row in rows) for metric in numeric}
    summary = json.loads((OUT / "summary.json").read_text())
    checks = 0
    for row in summary:
        if row["period"] != "warm" or row["group"] == "macro":
            continue
        for metric, value in computed[key(row)].items():
            assert abs(value - row[metric]) < 1e-10
            checks += 1
    references = {key(row): row for row in json.loads((OUT / "previous_reference_warm_summaries.json").read_text())}
    assert len(references) == 55 * 12
    references.update({identity: value for identity, value in computed.items()})
    assert len(references) == 61 * 12
    counts = json.loads((OUT / "regression_counts.json").read_text())
    failures = {}
    for expected in counts:
        actual = dict(wins=0, ties=0, regressions=0)
        for identity, row in computed.items():
            if identity[-2:] != (expected["candidate"], expected["variant"]):
                continue
            baseline = references[identity[:3] + (expected["reference"], expected["baseline_variant"])]
            for metric, direction in METRICS.items():
                value, old = primary_value(row, metric), primary_value(baseline, metric)
                gain = direction * (value - old)
                category = "wins" if gain > TOLERANCE else "regressions" if gain < -TOLERANCE else "ties"
                actual[category] += 1
                if category == "regressions":
                    failure_key = (expected["candidate"], expected["variant"], expected["reference"],
                                   expected["baseline_variant"], *identity[:3], metric)
                    failures[failure_key] = (value, old, gain)
        assert sum(actual.values()) == 96
        assert all(actual[name] == expected[name] for name in actual)
    assert len(counts) == 366
    saved = json.loads((OUT / "counterexamples.json").read_text())
    assert len(saved) == len(failures)
    for row in saved:
        identity = tuple(row[name] for name in ("candidate", "variant", "reference", "baseline_variant",
                                                "split", "group", "channel", "metric"))
        a, b, gain = failures.pop(identity)
        assert max(abs(row[name] - value) for name, value in (
            ("value", a), ("baseline", b), ("directional_gain", gain))) < 1e-10
    assert not failures

    trajectories = defaultdict(list)
    for row in cases:
        if row["split"] == "test" and row["start"] >= 20:
            trajectories[row["file"], row["channel"], row["map"], row["variant"]].append(row)
    paired = []
    for (filename, channel, name, mode), rows in sorted(trajectories.items()):
        if name != "warp":
            continue
        score = mean(row["standardized_interval_score"] for row in rows)
        for reference in ("frozen_best", "modulate"):
            old = trajectories[filename, channel, reference, mode]
            baseline = mean(row["standardized_interval_score"] for row in old)
            paired.append(dict(file=filename, channel=channel, strategy=mode, reference=reference,
                               candidate="warp", score=score, baseline_score=baseline,
                               score_reduction=baseline - score,
                               relative_reduction=(baseline - score) / baseline,
                               improved=score < baseline - TOLERANCE))
    assert len(paired) == 96
    (OUT / "paired_trajectory_scores.json").write_text(json.dumps(paired, indent=2) + "\n")
    paired_counts = {}
    for mode in MODES:
        for reference in ("frozen_best", "modulate"):
            selected = [row for row in paired if row["strategy"] == mode and row["reference"] == reference]
            assert len(selected) == 24
            paired_counts[mode + "/" + reference] = dict(improved=sum(row["improved"] for row in selected), total=24)
    report = dict(passed=True, case_rows=len(cases), numeric_group_checks=checks,
                  comparisons=len(counts), cells_per_comparison=96, tolerance=TOLERANCE,
                  all_counterexamples_verified=len(saved), point_forecast_mse_unchanged=True,
                  all55_old_reference_pairs_preserved=True, paired_trajectory_scores=paired_counts,
                  caveat="Exploratory repeatedly examined historical data; not independent statistical confirmation or formal coverage.")
    (OUT / "analysis_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
