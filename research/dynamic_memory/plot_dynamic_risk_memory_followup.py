"""Plot every selected96-cell outcome and all paired test-trajectory scores."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from analyze_dynamic_risk_memory_followup import OUT, METRICS, TOLERANCE, key, primary_value


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    current = {key(row): row for row in summary if row["period"] == "warm" and row["group"] != "macro"}
    references = {key(row): row for row in json.loads((OUT / "previous_reference_warm_summaries.json").read_text())}
    references.update(current)
    identities = sorted({identity[:3] for identity in current})
    assert len(identities) == 12
    comparisons = (
        ("memory_dynamic", "density_quantile", "memory_dynamic"),
        ("memory_dynamic", "modulate", "memory_dynamic"),
        ("memory_dynamic", "full", "memory_dynamic"),
        ("scale_fastcal", "separate_tails", "scale_fastcal"),
        ("scale_fastcal", "previous_2d", "global_smooth"),
    )
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), layout="constrained")
    values = []
    cmap = ListedColormap(["#ca604e", "#ededed", "#459e87"])
    for ax, (mode, reference, old_mode) in zip(axes, comparisons):
        cells = np.zeros((8, 12))
        for i, (metric, direction) in enumerate(METRICS.items()):
            for j, identity in enumerate(identities):
                a = primary_value(current[identity + ("warp", mode)], metric)
                b = primary_value(references[identity + (reference, old_mode)], metric)
                gain = direction * (a - b)
                category = 1 if gain > TOLERANCE else -1 if gain < -TOLERANCE else 0
                cells[i, j] = category
                ax.text(j, i, "+" if category > 0 else "-" if category < 0 else "=",
                        ha="center", va="center", color="white" if category else "black")
                values.append(dict(candidate="warp", strategy=mode, reference=reference,
                                   baseline_strategy=old_mode, split=identity[0], group=identity[1],
                                   channel=identity[2], metric=metric, value=a, baseline=b,
                                   directional_gain=gain, outcome=int(category)))
        ax.imshow(cells, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(8), list(METRICS))
        ax.set_xticks(range(12), [f"{split[:3]} {group} {channel}" for split, group, channel in identities], rotation=25)
        ax.set_title(f"warp/{mode} vs {reference}/{old_mode}: "
                     f"{int((cells > 0).sum())} improved, {int((cells == 0).sum())} tied, "
                     f"{int((cells < 0).sum())} regressed")
    fig.suptitle("All eight metrics and all12 Re/channel groups; tolerance1e-6\n"
                 "Exploratory reused data; no all-line or formal coverage claim")
    fig.savefig(OUT / "dynamic_memory_96cells.png", dpi=160)
    plt.close(fig)
    assert len(values) == 480
    with (OUT / "selected_primary_cell_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    paired = json.loads((OUT / "paired_trajectory_scores.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, reference in zip(axes, ("frozen_best", "modulate")):
        rows = [row for row in paired if row["strategy"] == "memory_dynamic" and row["reference"] == reference]
        assert len(rows) == 24
        gains = np.array([100 * row["relative_reduction"] for row in rows])
        ax.bar(range(24), gains, color=np.where(gains > 0, "#459e87", "#ca604e"))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(24), [row["file"].replace(".h5", "") + " " + row["channel"] for row in rows], rotation=90)
        ax.set_ylabel("Mean interval score reduction (%)")
        ax.set_title(f"warp/memory_dynamic vs {reference}: all24 test cases")
    fig.savefig(OUT / "dynamic_memory_paired_scores.png", dpi=160)
    plt.close(fig)
    print("DYNAMIC_MEMORY_PLOTS_WRITTEN", len(values), flush=True)


if __name__ == "__main__":
    main()
