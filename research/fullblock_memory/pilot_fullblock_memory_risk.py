"""Train-ID later-time test of actual four-call residual-risk memory inputs."""

import json
import time
from statistics import mean

import torch

from clean_reliability_common import ROOT, configure, dump, digest
from experiment_fullblock_memory_risk import OUT as TRAIN, NAMES, Provider, load
from experiment_reliability_transport_e12 import normalization, geometry
from diagnose_train_distribution_drift import OUT as DATA
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from direct_error_uncertainty import forecast_features
from reliability_risk_transport import base_features
from experiment_temporal_error_uncertainty import normalized
from experiment_continue_rich_uncertainty import MODES, controller
from conditional_online_risk import risk_features
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking, summarize


OUT = ROOT / "runs/pilot_fullblock_memory_risk_20261109"


@torch.no_grad()
def main():
    configure()
    assert (TRAIN / "complete.json").exists()
    OUT.mkdir(exist_ok=False)
    (OUT / "known").mkdir()
    data_protocol = json.loads((DATA / "protocol.json").read_text())
    records = data_protocol["records"]
    norm = normalization()
    provider = Provider(norm)
    geo = geometry(records)
    forecaster = StructuredFourStepPredictor(geo)
    initial = load(TRAIN / "calibration.pt")
    states = {name: {mode: controller(mode, initial[name]["separable_integral"], len(records))
                     for mode in MODES} for name in NAMES}
    dump(OUT / "protocol.json", dict(time=time.time(), train_protocol_sha256=digest(TRAIN / "protocol.json"),
        source_sha256=digest(ROOT / "pilot_fullblock_memory_risk.py"), records=records, starts=data_protocol["starts"],
        heads=NAMES, modes=MODES, cold_prelude=580, warm_queries=list(range(600,761,20)),
        gate="At least one last/full source under one fixed controller improves actualScore and top10energy versus matchednone by1e-6, with no worse meanabsoluteRe/channelcoverageerror by1e-6. Full-history advantage overlast checked separately. No gain or method chosen on dev/test.",
        causality="Original4nativecalls perknownhistory,all20futureframesjointly. Actual point forecast and last48features regenerated; point and frozenriskbounds must exactly match archived known packets. Memory equals currentknownhistory minus actual previousissued point; known packet saved before currentlabel read.",
        comparison="Three matched48340parameter heads and frozen45060reference, same two originalonlinecontrollers. Neural fitting ended600, later times reused sameID diagnostic, independent10IDcalibration only initializes.",
        limits="Not fresh independent confirmation. All primary metrics retained; pilot gate is not the final96cell success claim."))
    rows = []
    previous_point = previous_residual = None
    for step, start in enumerate(data_protocol["starts"]):
        source = load(DATA / "known" / (str(start) + ".pt"))
        observed = source["observed"]
        h = (observed - norm["mean"]) / norm["std"]
        calls = []
        hook = forecaster.model.net.register_forward_pre_hook(lambda module, args: calls.append(int(args[1][0])))
        try:
            point, features = forecast_features(forecaster.model, h)
        finally:
            hook.remove()
        assert calls == [99, 66, 33, 0]
        assert torch.equal(point, source["prediction"])
        raw_last = features[:, -48:].half()
        base = base_features(observed, point * norm["std"] + norm["mean"], norm["mean"], norm["std"], geo)
        base, last = normalized(base, raw_last.float(), provider.stats)
        memory = torch.zeros_like(h) if previous_point is None else h - previous_point
        if previous_residual is not None:
            assert torch.equal(memory, previous_residual)
        bounds = {"frozen_rich": provider.frozen.bounds(base, last, h, torch.ones_like(point[:1]))}
        assert torch.equal(bounds["frozen_rich"][0], source["rich_lower"])
        assert torch.equal(bounds["frozen_rich"][1], source["rich_upper"])
        for name, head in provider.heads.items():
            bounds[name] = head.bounds(base, last, h, memory, previous_point is not None)
        issued = {name: {mode: state.issue(*bounds[name], risk_features(h, point, .5 * (bounds[name][1] - bounds[name][0])))
                         for mode, state in modes.items()} for name, modes in states.items()}
        torch.save(dict(prediction=point.cpu(), last=raw_last.cpu(), previous_error=memory.cpu(),
            available=previous_point is not None, native_labels=calls), OUT / "known" / (str(start) + ".pt"))
        with (OUT / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="issued", start=start, time=time.time(), native_labels=calls,
                prior_complete_windows=step)) + "\n")
        residual = load(DATA / "labels" / (str(start) + ".pt"))
        with (OUT / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="revealed", start=start, time=time.time())) + "\n")
        mse = (residual * norm["std"]).square().mean((2, 3, 4))
        for name, modes in issued.items():
            raw = {"raw_" + key: value for key, value in ranking(.5 * (bounds[name][1] - bounds[name][0]), residual.abs()).items()}
            for mode, (lo, hi) in modes.items():
                values = dict(**metrics(lo, hi, residual), **ranking(.5 * (hi - lo), residual.abs()), **raw)
                for i, record in enumerate(records):
                    for c, channel in enumerate(("u", "v")):
                        rows.append(dict(kind="diffusion", split="train_time_pilot", group=record["group"], file=record["file"],
                            start=start, period="later" if start >= 700 else "post_cold" if start >= 600 else "prefix",
                            channel=channel, map=name, variant=mode, mse=float(mse[i,c]),
                            **{key: float(value[i,c]) for key,value in values.items()}))
                states[name][mode].reveal(residual)
        previous_point = point.clone()
        previous_residual = residual.clone()
        print("FULLBLOCK_MEMORY_PILOT", start, flush=True)
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", summarize(rows))
    results = {}
    for mode in MODES:
        values = {}
        for name in NAMES:
            # Use all eight post-cold windows, not only the final four.
            cases = [r for r in rows if r["map"] == name and r["variant"] == mode and r["start"] >= 600]
            groups = {}
            for row in cases:
                groups.setdefault((row["group"],row["channel"]),[]).append(row)
            values[name] = dict(score=mean(mean(r["standardized_interval_score"] for r in group) for group in groups.values()),
                coverage_error=mean(abs(mean(r["coverage"] for r in group)-.9) for group in groups.values()),
                top10=mean(mean(r["top10_energy"] for r in group) for group in groups.values()))
        passed = []
        for name in ("last", "full"):
            new, old = values[name], values["none"]
            if new["score"] < old["score"] - 1e-6 and new["top10"] > old["top10"] + 1e-6 and new["coverage_error"] <= old["coverage_error"] + 1e-6:
                passed.append(name)
        full_vs_last = (values["full"]["score"] < values["last"]["score"] - 1e-6
                        and values["full"]["top10"] > values["last"]["top10"] + 1e-6)
        results[mode] = dict(values=values, passed_memory_modes=passed, full_history_beats_last_on_score_and_ranking=full_vs_last)
    dump(OUT / "gate.json", dict(results=results, proceed_full_evaluation=any(row["passed_memory_modes"] for row in results.values())))
    dump(OUT / "complete.json", dict(time=time.time(), actual_native_forecasts=440, full_issue_reveal_pairs=10,
        actual_previous_error_equals_last_revealed_residual=True, original_points_and_frozen_risk_bounds_exact=True))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
