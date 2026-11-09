"""Recompute aggregate and paired improvements from the exported frozen metrics.

No model weights or original dataset are needed. Run: python analyze_results.py
Positive gain_percent means lower error than the control.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "analysis"
CASE = ("split", "period", "group", "file", "start", "channel")
CONTROLS = ("physical_state_cnn", "physical_delta_cnn", "state_fourstep",
            "delta_fourstep", "physical_persistence", "physical_state_diffusion")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (OUT / "per_case.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    buckets = defaultdict(list)
    for row in rows:
        case = tuple(row[k] for k in CASE)
        key = case + (row["method"],)
        assert key not in lookup, f"Duplicate case: {key}"
        lookup[key] = row
        buckets[(row["split"], row["period"], row["group"], row["method"])].append(row)

    grouped = []
    for (split, period, group, method), values in sorted(buckets.items()):
        grouped.append(dict(split=split, period=period, group=group, method=method,
                            mse=mean(float(v["mse"]) for v in values),
                            standardized_mse=mean(float(v["standardized_mse"]) for v in values)))
    macro = defaultdict(list)
    for row in grouped:
        macro[(row["split"], row["period"], row["method"])].append(row)
    for (split, period, method), values in sorted(macro.items()):
        grouped.append(dict(split=split, period=period, group="macro", method=method,
                            mse=mean(v["mse"] for v in values),
                            standardized_mse=mean(v["standardized_mse"] for v in values)))
    write_csv(OUT / "recomputed_group_metrics.csv", grouped)

    # Every matched window/channel is retained, including deteriorations.
    paired = []
    for key, candidate in sorted(lookup.items()):
        if key[-1] != "physical_state_zero":
            continue
        for control in CONTROLS:
            reference = lookup[key[:-1] + (control,)]
            a, b = float(candidate["mse"]), float(reference["mse"])
            paired.append(dict(zip(CASE, key[:-1]), candidate=key[-1], control=control,
                               candidate_mse=a, control_mse=b,
                               mse_difference=a - b, gain_percent=100 * (1 - a / b)))
    write_csv(OUT / "paired_windows.csv", paired)

    # Check the recomputed values against both existing published group tables.
    checked = 0
    for row in grouped:
        folder = "deterministic_fourstep" if row["method"] in ("state_fourstep", "delta_fourstep") else "physical_direct20"
        with (ROOT / "results" / folder / "point_references.csv").open(newline="") as stream:
            references = list(csv.DictReader(stream))
        matches = [r for r in references if all(str(r[k]) == str(row[k]) for k in ("split", "period", "group", "method"))]
        assert len(matches) == 1, row
        for metric in ("mse", "standardized_mse"):
            assert abs(row[metric] - float(matches[0][metric])) <= 1e-10 * max(1, abs(row[metric])), (row, metric)
        checked += 1

    table = {(r["split"], r["period"], r["group"], r["method"]): r for r in grouped}
    contrasts = []
    for split in ("development", "test"):
        for period in ("main", "confirmation"):
            a = table[(split, period, "macro", "physical_state_zero")]
            for control in CONTROLS:
                b = table[(split, period, "macro", control)]
                cases = [r for r in paired if r["split"] == split and r["period"] == period and r["control"] == control]
                contrasts.append(dict(split=split, period=period, control=control,
                                      candidate_mse=a["mse"], control_mse=b["mse"],
                                      gain_percent=100 * (1 - a["mse"] / b["mse"]),
                                      standardized_gain_percent=100 * (1 - a["standardized_mse"] / b["standardized_mse"]),
                                      window_channel_win_fraction=mean(r["mse_difference"] < 0 for r in cases)))
    write_csv(OUT / "macro_improvements.csv", contrasts)
    report = dict(source_rows=len(rows), paired_rows=len(paired), checked_group_rows=checked,
                  passed=True, caveat="Window/channel rows are correlated; win fractions are descriptive, not independent statistical evidence.")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
