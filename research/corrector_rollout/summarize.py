"""Trajectory-weighted scores, paired bootstrap intervals, and static figures."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector.run import table

ORACLE = "oracle_one_step"
ROLLOUT = "oracle_initialized_free_rollout"
METRICS = ("residual_mse", "physical_mse")
COLORS = {
    "Persistence": "#737b8b",
    "R-only": "#2364aa",
    "State-conditioned": "#ad4f16",
    "Uncorrected DiPred": "#252a33",
}


def average(rows):
    return {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}


def per_trajectory(rows):
    grouped = defaultdict(list)
    fields = ("experiment", "split", "trajectory", "re_group", "re", "aoa", "lead", "channel", "model")
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, windows in grouped.items():
        assert {w["history_start"] for w in windows} == {400, 420} and len(windows) == 2
        output.append(dict(zip(fields, key), windows=2, members_per_window=32, **average(windows)))
    return output


def aggregate_trajectory(rows):
    groups = defaultdict(list)
    for row in rows:
        for re_group in ("all", row["re_group"]):
            for lead in ("2–20", row["lead"]):
                for channel in ("uv", row["channel"]):
                    key = row["split"], re_group, lead, channel, row["experiment"], row["model"]
                    groups[key, row["trajectory"]].append(row)
    grouped = defaultdict(dict)
    for (key, trajectory), scores in groups.items():
        expected = (19 if key[2] == "2–20" else 1) * (2 if key[3] == "uv" else 1)
        assert len(scores) == expected
        grouped[key][trajectory] = average(scores)
    return grouped


def interval(values):
    return np.quantile(values, (0.025, 0.975)).tolist()


def pair_stats(model, baseline, indices, relative=True):
    diff = model - baseline
    diff_low, diff_high = interval(diff[indices].mean(axis=1))
    output = dict(model_value=model.mean(), baseline_value=baseline.mean(), delta=diff.mean(),
                  delta_ci_low=diff_low, delta_ci_high=diff_high,
                  lower_count=int((diff < 0).sum()), equal_count=int((diff == 0).sum()))
    if relative:
        assert (baseline > 0).all()
        lo, hi = interval(100 * (1 - model[indices].mean(axis=1) / baseline[indices].mean(axis=1)))
        output.update(improvement_pct=100 * (1 - model.mean() / baseline.mean()),
                      improvement_ci_low_pct=lo, improvement_ci_high_pct=hi)
    return output


def bootstrap_indices(n):
    return np.random.default_rng(42).integers(n, size=(10000, n))


def compare(grouped):
    metrics = []
    comparisons = []
    for (split, re_group, lead, channel, experiment, model), values in grouped.items():
        scope = dict(split=split, re_group=re_group, lead=lead, channel=channel,
                     experiment=experiment, model=model, trajectories=len(values))
        files = sorted(values)
        res = np.array([values[file]["residual_mse"] for file in files])
        field = np.array([values[file]["physical_mse"] for file in files])
        metrics.append(dict(**scope, residual_mse=res.mean(), physical_mse=field.mean()))
        baselines = []
        if model != "Persistence":
            baselines.append((experiment, "Persistence"))
        if model == "State-conditioned":
            baselines.append((experiment, "R-only"))
        if experiment == ROLLOUT and model != "Uncorrected DiPred":
            baselines.extend(((ROLLOUT, "Uncorrected DiPred"), (ORACLE, model)))
        indices = bootstrap_indices(len(files))
        for baseline_experiment, baseline in baselines:
            other = grouped[split, re_group, lead, channel, baseline_experiment, baseline]
            assert set(other) == set(values)
            ref = np.array([other[file]["residual_mse"] for file in files])
            comparisons.append(dict(**scope, baseline_experiment=baseline_experiment,
                                    baseline_model=baseline, **pair_stats(res, ref, indices)))
    return metrics, comparisons


def stability(grouped):
    rows = []
    for (split, re_group, lead, channel, experiment, model), early in grouped.items():
        if lead != 2 or experiment != ROLLOUT or model != "State-conditioned":
            continue
        files = sorted(early)

        def growth(name):
            start = grouped[split, re_group, 2, channel, ROLLOUT, name]
            end = grouped[split, re_group, 20, channel, ROLLOUT, name]
            assert set(start) == set(end) == set(files)
            return np.array([end[f]["residual_mse"] - start[f]["residual_mse"] for f in files])

        state_growth = growth(model)
        residual_growth = growth("R-only")
        rows.append(dict(split=split, re_group=re_group, channel=channel, lead_from=2, lead_to=20,
                         model=model, baseline_model="R-only", trajectories=len(files),
                         **pair_stats(state_growth, residual_growth, bootstrap_indices(len(files)),
                                      relative=False)))
    return rows


def draw_curves(out, metrics, comparisons):
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    leads = np.arange(2, 21)
    select = dict(split="test", re_group="all", channel="uv")

    def curve(rows, **filters):
        selected = [r for r in rows if isinstance(r["lead"], int)
                    and all(r[key] == value for key, value in dict(select, **filters).items())]
        selected.sort(key=lambda r: r["lead"])
        np.testing.assert_array_equal([r["lead"] for r in selected], leads)
        return selected

    fig, ax = plt.subplots(figsize=(7.4, 4.3), layout="constrained")
    for model, color in COLORS.items():
        free = curve(metrics, experiment=ROLLOUT, model=model)
        ax.plot(leads, [r["residual_mse"] for r in free], color=color, lw=2, label=model)
        if model != "Uncorrected DiPred":
            oracle = curve(metrics, experiment=ORACLE, model=model)
            ax.plot(leads, [r["residual_mse"] for r in oracle], color=color, lw=1, ls="--")
    ax.plot([], [], "--", color="#333333", lw=1, label="Dashed: oracle one-step")
    ax.set(xlabel="Physical lead", ylabel="Residual prediction MSE", xticks=[2, 5, 10, 15, 20],
           title="Test trajectories · u/v average")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "mse_by_lead.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3), layout="constrained")
    for experiment, label, style in [(ROLLOUT, "Free rollout", "-"), (ORACLE, "Oracle one-step", "--")]:
        rows = curve(comparisons, experiment=experiment, baseline_experiment=experiment,
                     model="State-conditioned", baseline_model="R-only")
        color = COLORS["State-conditioned"] if experiment == ROLLOUT else COLORS["R-only"]
        ax.fill_between(leads, [r["improvement_ci_low_pct"] for r in rows],
                        [r["improvement_ci_high_pct"] for r in rows], color=color, alpha=0.12)
        ax.plot(leads, [r["improvement_pct"] for r in rows], style, lw=2, color=color, label=label)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set(xlabel="Physical lead", ylabel="MSE reduction vs R-only (%)",
           xticks=[2, 5, 10, 15, 20], title="State-conditioned · test trajectories")
    ax.legend(frameon=False, title="95% paired trajectory intervals")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "state_vs_r_only.png")
    plt.close(fig)


def draw_spatial(out, display, case):
    split, trajectory, start, seed = case
    leads = (5, 20)
    fig, axes = plt.subplots(4, 3, figsize=(9, 6.9), layout="constrained")
    for channel_index, channel in enumerate(("u", "v")):
        channel_axes = axes[2 * channel_index:2 * channel_index + 2]
        limit = max(np.abs(window[channel_index, np.array(leads) - 2]).max()
                    for window in display.values())
        for row, lead in zip(channel_axes, leads):
            for ax, (name, frames) in zip(row, display.items()):
                img = ax.imshow(frames[channel_index, lead - 2], origin="lower",
                                cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set(xticks=[], yticks=[])
                if channel_index == 0 and lead == leads[0]:
                    ax.set_title(name, fontsize=10)
            row[0].set_ylabel(f"{channel} · lead {lead}")
        fig.colorbar(img, ax=channel_axes, shrink=0.85, label=f"{channel} residual")
    fig.suptitle(f"{split.capitalize()} · {trajectory} · history start {start} · seed {seed}")
    fig.savefig(out / "fixed_trajectory_residuals.png")
    plt.close(fig)


def write_results(out, window_rows, display, case):
    trajectories = per_trajectory(window_rows)
    grouped = aggregate_trajectory(trajectories)
    metrics, comparisons = compare(grouped)
    table(out / "per_trajectory.csv", trajectories)
    table(out / "scores.csv", metrics)
    table(out / "paired_comparisons.csv", comparisons)
    table(out / "stability.csv", stability(grouped))
    draw_curves(out, metrics, comparisons)
    draw_spatial(out, display, case)
