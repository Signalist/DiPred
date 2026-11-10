"""Plot complete 96-cell comparisons, retaining every regression.

Requires matplotlib and numpy; the CSV analysis itself uses only stdlib.
Colors classify directional gain using the unchanged 1e-6 tolerance.
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from analyze_continued_reliability import OUT, METRICS, TOLERANCE, key, primary_value


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    current = {key(row): row for row in summary
               if row["period"] == "warm" and row["group"] != "macro"}
    references = {key(row): row for row in json.loads(
        (OUT / "reference_warm_summaries.json").read_text())}
    comparisons = (
        ("memory_dynamic", "full", "memory_dynamic"),
        ("memory_dynamic", "quantile", "memory_dynamic"),
        ("memory_dynamic", "previous_2d", "global_smooth"),
        ("scale_fastcal", "previous_2d", "global_smooth"),
    )
    identities = sorted({identity[:3] for identity in current})
    assert len(identities) == 12
    fig, axes = plt.subplots(4, 1, figsize=(14, 15), layout="constrained")
    values = []
    cmap = ListedColormap(["#ca604e", "#ededed", "#459e87"])
    for ax, (strategy, reference, baseline_strategy) in zip(axes, comparisons):
        cells = np.zeros((len(METRICS), len(identities)))
        for i, (metric, direction) in enumerate(METRICS.items()):
            for j, identity in enumerate(identities):
                a = primary_value(current[identity + ("density_quantile", strategy)], metric)
                b = primary_value(references[identity + (reference, baseline_strategy)], metric)
                gain = direction * (a - b)
                category = 1 if gain > TOLERANCE else -1 if gain < -TOLERANCE else 0
                cells[i, j] = category
                ax.text(j, i, "+" if category == 1 else "-" if category == -1 else "=",
                        ha="center", va="center", color="white" if category else "black")
                values.append(dict(
                    candidate="density_quantile", strategy=strategy,
                    reference=reference, baseline_strategy=baseline_strategy,
                    split=identity[0], group=identity[1], channel=identity[2],
                    metric=metric, value=a, baseline=b, directional_gain=gain,
                    outcome="improved" if category == 1 else "regressed" if category == -1 else "tied",
                ))
        ax.imshow(cells, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(METRICS)), list(METRICS))
        ax.set_xticks(range(12), [f"{s[:3]} {g} {c}" for s, g, c in identities], rotation=25)
        ax.set_title(f"density_quantile/{strategy} vs {reference}/{baseline_strategy}: "
                     f"{int((cells > 0).sum())} improved, {int((cells == 0).sum())} tied, "
                     f"{int((cells < 0).sum())} regressed")
    fig.suptitle("All eight metrics, all 12 Re/channel cells; tolerance 1e-6\n"
                 "Exploratory reused data; no full-line improvement or formal coverage claim")
    fig.savefig(OUT / "primary_comparison_cells.png", dpi=160)
    plt.close(fig)
    assert len(values) == 384
    with (OUT / "selected_primary_cell_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    print("PLOTTED_ALL_SELECTED_CELLS", len(values))


if __name__ == "__main__":
    main()
