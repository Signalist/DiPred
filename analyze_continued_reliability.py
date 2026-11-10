"""Recompute all 440 continued-head comparisons using only the standard library.

All development/test windows and all eight primary metrics are retained.
This checks the exported evidence, not statistical independence or guarantees.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / "results/reliability/continued_quantile"
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


def paired_trajectory_scores(cases):
    """Keep every test identity/channel, including worse individual cases."""
    current = [row for row in cases if row["split"] == "test"
               and int(row["start"]) >= 20 and row["variant"] == "memory_dynamic"]
    with (OUT.parent / "fullblock_memory/per_case.csv").open(newline="") as stream:
        published = [row for row in csv.DictReader(stream) if row["split"] == "test"
                     and int(row["start"]) >= 20 and row["map"] == "full"
                     and row["variant"] == "memory_dynamic"]
    buckets = defaultdict(list)
    for row in current + published:
        if row["map"] in ("full", "density_reference", "density_quantile"):
            buckets[row["map"], row["file"], row["channel"]].append(row)
    output = []
    for head, filename, channel in sorted(buckets):
        if head != "density_quantile":
            continue
        rows = buckets[head, filename, channel]
        assert len(rows) == 38
        score = mean(float(row["standardized_interval_score"]) for row in rows)
        for reference in ("full", "density_reference"):
            old = buckets[reference, filename, channel]
            assert len(old) == 38
            baseline = mean(float(row["standardized_interval_score"]) for row in old)
            output.append(dict(
                file=filename, channel=channel, reference=reference,
                candidate="density_quantile", strategy="memory_dynamic",
                interval_score=score, baseline_score=baseline,
                score_reduction=baseline - score,
                relative_score_reduction=(baseline - score) / baseline,
                improved=score < baseline - TOLERANCE,
            ))
    assert len(output) == 48
    with (OUT / "paired_trajectory_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return {reference: dict(
        improved=sum(row["improved"] for row in output if row["reference"] == reference),
        total=24,
    ) for reference in ("full", "density_reference")}


def main():
    with (OUT / "per_case.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        cases = list(reader)
    assert len(cases) == 14976
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
    assert len(references) == 55 * 12
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
    assert len(counts) == 440
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
    for head in ("density_reference", "density_quantile", "density_gaussian", "density_student4"):
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
    paired = paired_trajectory_scores(cases)
    report = dict(passed=True, case_rows=len(cases), numeric_group_checks=numeric_checks,
                  comparisons=len(counts), cells_per_comparison=96, tolerance=TOLERANCE,
                  all_counterexamples_verified=len(saved_failures), point_forecasts_unchanged=True,
                  test_warm_macro=macros,
                  paired_test_trajectory_scores=paired,
                  caveat="Repeatedly examined historical data; no independent confirmation or formal joint coverage.")
    (OUT / "analysis_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
