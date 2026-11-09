"""Recompute current reliability summaries and96-cell regressions, no GPU/data.

Uses only Python's standard library and exported complete window metrics.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / "results/reliability"
METRICS = {"standardized_interval_score": -1, "standardized_width": -1,
           "top10_energy": 1, "per_frame_top10_energy": 1, "relative_aurc": -1,
           "worst_frame_coverage": 1, "worst_patch_coverage": 1, "coverage_error": -1}
TOLERANCE = 1e-6


def main():
    with (OUT / "per_case.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        cases = list(reader)
    identifiers = {"kind", "split", "file", "group", "period", "channel", "map", "variant", "method", "start"}
    metrics = [name for name in reader.fieldnames if name not in identifiers]
    buckets = defaultdict(list)
    unique = set()
    for row in cases:
        key = tuple(row[k] for k in ("split", "file", "start", "channel", "method"))
        assert key not in unique
        unique.add(key)
        for period in (row["period"], "warm" if int(row["start"]) >= 20 else "cold"):
            buckets[(row["split"], period, row["group"], row["channel"], row["method"])].append(row)
    recomputed = {}
    for key, rows in buckets.items():
        # The newer experiments did not record the optional pre-calibration
        # raw_* ranking diagnostics. Blank means unmeasured, never zero.
        recomputed[key] = {metric: mean(float(r[metric]) for r in rows) for metric in metrics
                           if all(r[metric] != "" for r in rows)}
    saved = json.loads((OUT / "summary.json").read_text())
    checked = 0
    for row in saved:
        if row["group"] == "macro":
            continue
        key = tuple(row[k] for k in ("split", "period", "group", "channel", "method"))
        for metric, value in recomputed[key].items():
            assert abs(row[metric] - value) < 1e-10, (key, metric)
            checked += 1
    counts, failures = [], []
    references = ("previous_2d", "rich_scale", "tied_tails", "low_regression", "symmetric_denoising",
                  "best_mean_score", "context_mean_score_base", "context_mean_score_hold",
                  "context_mean_score_drift", "context_balanced_base", "context_balanced_hold",
                  "context_balanced_drift")
    for candidate in ("low_regression", "best_mean_score", "context_mean_score", "context_balanced"):
        for reference in references:
            count = dict(candidate=candidate, reference=reference, wins=0, ties=0, regressions=0)
            for key, row in recomputed.items():
                split, period, group, channel, method = key
                if period != "warm" or method != candidate:
                    continue
                old = recomputed[(split, period, group, channel, reference)]
                for metric, direction in METRICS.items():
                    a = abs(row["coverage"] - .9) if metric == "coverage_error" else row[metric]
                    b = abs(old["coverage"] - .9) if metric == "coverage_error" else old[metric]
                    gain = direction * (a - b)
                    state = "wins" if gain > TOLERANCE else "regressions" if gain < -TOLERANCE else "ties"
                    count[state] += 1
                    if state == "regressions":
                        failures.append(dict(candidate=candidate, reference=reference, split=split,
                            group=group, channel=channel, metric=metric, value=a, baseline=b, directional_gain=gain))
            assert sum(count[k] for k in ("wins", "ties", "regressions")) == 96
            counts.append(count)
    with (OUT / "current_counterexamples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failures[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(failures)
    (OUT / "current_regression_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    report = dict(passed=True, case_rows=len(cases), checked_numeric_group_metrics=checked,
                  primary_cells_per_comparison=96, tolerance=TOLERANCE,
                  caveat="Exploratory reused trajectories; windows/pixels correlated, no independent confirmation or coverage guarantee.")
    # Coverage deviations must be taken per Re/channel before averaging;
    # abs(grand-mean coverage - .9) would hide cancellation between groups.
    report["test_warm_macro"] = {}
    for candidate in ("low_regression", "best_mean_score", "context_mean_score", "context_balanced"):
        rows = [row for key, row in recomputed.items()
                if key[0] == "test" and key[1] == "warm" and key[4] == candidate]
        report["test_warm_macro"][candidate] = dict(
            coverage=mean(row["coverage"] for row in rows),
            interval_score=mean(row["standardized_interval_score"] for row in rows),
            interval_width=mean(row["standardized_width"] for row in rows),
            mean_absolute_group_coverage_error=mean(abs(row["coverage"] - .9) for row in rows))
    (OUT / "analysis_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
