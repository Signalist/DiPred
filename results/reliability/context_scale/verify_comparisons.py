"""Recompute all160 source comparisons using only exported group summaries.

From the repository root:
    python results/reliability/context_scale/verify_comparisons.py

The main analyze_reliability.py separately reconstructs group metrics from
all33,696 window records. This script additionally covers all20 original
strong references, including historical candidates not deployed in this repo.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = {"standardized_interval_score": -1, "standardized_width": -1,
           "top10_energy": 1, "per_frame_top10_energy": 1, "relative_aurc": -1,
           "worst_frame_coverage": 1, "worst_patch_coverage": 1, "coverage_error": -1}
TOLERANCE = 1e-6


def main():
    rows = json.loads((ROOT / "summary.json").read_text())
    refs = json.loads((ROOT / "strong_reference_summary.json").read_text())
    def key(row):
        return tuple(row[k] for k in ("split", "group", "channel", "map", "variant"))
    references = {key(row): row for row in refs}
    expected = json.loads((ROOT / "regression_counts.json").read_text())
    failures = {}
    checked = 0
    for comparison in expected:
        counts = dict(wins=0, ties=0, regressions=0)
        for row in rows:
            if (row["period"] != "warm" or row["group"] == "macro"
                    or (row["map"], row["variant"]) != (comparison["candidate"], comparison["variant"])):
                continue
            ref = references[(row["split"], row["group"], row["channel"],
                              comparison["reference"], comparison["baseline_variant"])]
            for metric, direction in METRICS.items():
                a = abs(row["coverage"] - .9) if metric == "coverage_error" else row[metric]
                b = abs(ref["coverage"] - .9) if metric == "coverage_error" else ref[metric]
                gain = direction * (a - b)
                category = "wins" if gain > TOLERANCE else "regressions" if gain < -TOLERANCE else "ties"
                counts[category] += 1
                if category == "regressions":
                    identity = (comparison["candidate"], comparison["variant"], comparison["reference"],
                                comparison["baseline_variant"], row["split"], row["group"], row["channel"], metric)
                    failures[identity] = (a, b, gain)
        assert sum(counts.values()) == 96
        assert all(counts[k] == comparison[k] for k in counts), (comparison, counts)
        checked += 96
    saved = json.loads((ROOT / "counterexamples.json").read_text())
    assert len(saved) == len(failures)
    for row in saved:
        identity = tuple(row[k] for k in ("candidate", "variant", "reference", "baseline_variant",
                                         "split", "group", "channel", "metric"))
        assert all(abs(a - b) < 1e-12 for a, b in zip(failures[identity],
                   (row["value"], row["baseline"], row["directional_gain"])))
    report = dict(passed=True, comparisons=len(expected), primary_cells_checked=checked,
                  counterexamples_recomputed=len(failures), tolerance=TOLERANCE,
                  limit="Deterministic evidence reanalysis, not an independent statistical confirmation.")
    (ROOT / "strong_reference_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
