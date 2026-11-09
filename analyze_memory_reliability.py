"""Recompute all 390 memory-head comparisons using only the standard library.

All development/test windows and all eight primary metrics are retained.
This checks the exported evidence, not statistical independence or guarantees.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / "results/reliability/fullblock_memory"
METRICS = {
    "standardized_interval_score": -1,
    "standardized_width": -1,
    "top10_energy": 1,
    "per_frame_top10_energy": 1,
    "relative_aurc": -1,
    "worst_frame_coverage": 1,
    "worst_patch_coverage": 1,
    "coverage_error": -1,
}
TOLERANCE = 1e-6


def key(row):
    return tuple(row[k] for k in ("split", "group", "channel", "map", "variant"))


def primary_value(row, metric):
    return abs(row["coverage"] - 0.9) if metric == "coverage_error" else row[metric]


def main():
    with (OUT / "per_case.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        cases = list(reader)
    assert len(cases) == 18720
    identifiers = {"kind", "split", "file", "group", "start", "period", "channel", "map", "variant"}
    numeric = [name for name in reader.fieldnames if name not in identifiers]
    unique, points = set(), {}
    buckets = defaultdict(list)
    for row in cases:
        identity = tuple(row[k] for k in ("split", "file", "start", "channel", "map", "variant"))
        assert identity not in unique
        unique.add(identity)
        point_key = identity[:4]
        point_mse = float(row["mse"])
        assert point_mse == points.setdefault(point_key, point_mse)
        if int(row["start"]) >= 20:
            buckets[key(row)].append(row)
    recomputed = {}
    for identity, rows in buckets.items():
        # Re groups contain different trajectory counts. Each trajectory must
        # contribute all 38 warm windows, rather than imposing equal group size.
        assert set(Counter(r["file"] for r in rows).values()) == {38}
        recomputed[identity] = {metric: mean(float(r[metric]) for r in rows) for metric in numeric}

    saved = json.loads((OUT / "summary.json").read_text())
    numeric_checks = 0
    for row in saved:
        if row["period"] != "warm" or row["group"] == "macro":
            continue
        for metric, value in recomputed[key(row)].items():
            assert abs(row[metric] - value) < 1e-10, (key(row), metric)
            numeric_checks += 1
    references = {key(row): row for row in json.loads((OUT / "reference_warm_summaries.json").read_text())}
    assert len(references) == 39 * 12
    counts = json.loads((OUT / "regression_counts.json").read_text())
    failures = {}
    for expected in counts:
        actual = dict(wins=0, ties=0, regressions=0)
        for identity, row in recomputed.items():
            if identity[-2:] != (expected["candidate"], expected["variant"]):
                continue
            baseline = references[identity[:3] + (expected["reference"], expected["baseline_variant"])]
            for metric, direction in METRICS.items():
                a, b = primary_value(row, metric), primary_value(baseline, metric)
                gain = direction * (a - b)
                category = "wins" if gain > TOLERANCE else "regressions" if gain < -TOLERANCE else "ties"
                actual[category] += 1
                if category == "regressions":
                    fail_key = (expected["candidate"], expected["variant"], expected["reference"],
                                expected["baseline_variant"], *identity[:3], metric)
                    failures[fail_key] = (a, b, gain)
        assert sum(actual.values()) == 96
        assert all(actual[name] == expected[name] for name in actual), (expected, actual)
    assert len(counts) == 390
    with (OUT / "counterexamples.csv").open(newline="") as stream:
        saved_failures = list(csv.DictReader(stream))
    assert len(saved_failures) == len(failures)
    for row in saved_failures:
        identity = tuple(row[k] for k in ("candidate", "variant", "reference", "baseline_variant",
                                          "split", "group", "channel", "metric"))
        a, b, gain = failures.pop(identity)
        assert max(abs(float(row[name]) - value) for name, value in (
            ("value", a), ("baseline", b), ("directional_gain", gain))) < 1e-10
    assert not failures

    macros = {}
    for head in ("frozen_rich", "none", "last", "mean", "full"):
        for policy in ("memory_dynamic", "scale_fastcal"):
            rows = [row for identity, row in recomputed.items()
                    if identity[0] == "test" and identity[-2:] == (head, policy)]
            assert len(rows) == 6
            macros[head + "/" + policy] = dict(
                coverage=mean(r["coverage"] for r in rows),
                interval_score=mean(r["standardized_interval_score"] for r in rows),
                interval_width=mean(r["standardized_width"] for r in rows),
                # Take absolute deviations BEFORE averaging across Re/channel.
                mean_absolute_group_coverage_error=mean(abs(r["coverage"] - 0.9) for r in rows),
            )
    report = dict(passed=True, case_rows=len(cases), numeric_group_checks=numeric_checks,
                  comparisons=len(counts), cells_per_comparison=96, tolerance=TOLERANCE,
                  all_counterexamples_verified=len(saved_failures), point_forecasts_unchanged=True,
                  test_warm_macro=macros,
                  caveat="Repeatedly examined historical data; no independent confirmation or formal joint coverage.")
    (OUT / "analysis_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
