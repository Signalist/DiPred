"""Run the fixed diagnostic without invoking training or generating conditions."""

import argparse
import csv
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from research.residual_uncertainty_diffusion.core import ARMS, METRICS, crps, csvwrite, dump, read, state_hash, tensor_hash
from research.residual_uncertainty_diffusion.experiment import freeze_hash
from .evidence import Evidence, OUT, ROOT, input_hashes, select
from .sampler import LABELS, sample
from .tables import (
    AMPLITUDE, COMPONENT, CONDITION, DECOMPOSITION, decompose, fd_status, save_scores, score_rows,
)
from .tests import run_synthetic


class Diagnostic:
    def __init__(self, destination, legacy_precision=False, prior_groups=0):
        self.destination = destination
        self.legacy_precision = legacy_precision
        self.prior_groups = prior_groups
        self.started = perf_counter()
        self.counts = dict(complete_forward_groups=0, gradient_graphs=0, autograd_grad_calls=0,
                           optimizer_instances=0, optimizer_updates=0)
        self.forward_log = []
        self.gradient_log = []
        self.milestones = {}

    def checkpoint(self, name):
        self.milestones[name] = perf_counter() - self.started
        self.save_cost()

    def save_cost(self):
        dump(self.destination / "COST.json", dict(
            **self.counts, wall_seconds=perf_counter() - self.started,
            elapsed_at_milestones=self.milestones,
            member_count=32, steps_per_group=4, xpf_rebuilds=0, feedback_rebuilds=0,
            prior_complete_forward_groups=self.prior_groups,
            cumulative_complete_forward_groups=self.prior_groups + self.counts["complete_forward_groups"],
        ))
        csvwrite(self.destination / "FORWARDS.csv", self.forward_log)
        csvwrite(self.destination / "GRADIENT_REQUESTS.csv", self.gradient_log)

    def prepare(self):
        records, self.stage, self.fd_keys = select(self.destination)
        self.meta = {r["cache_key"]: {k: r[k] for k in ("split", "trajectory_id", "Re", "start")} for r in records}
        self.evidence = Evidence()
        self.native, self.norm, self.outs, self.corrections = self.evidence.load_models(matmul_tf32=self.legacy_precision)
        self.inputs, self.targets, conditions, self.poisoned = self.evidence.load_records(records, self.norm)
        self.original_condition_hashes = {r["cache_key"]: r["input_sha256"] for r in conditions}
        dump(self.destination / "CONDITION_NOISE_HASHES.json", conditions)
        dump(self.destination / "NORMALIZATION.json", {
            "condition_scales": self.evidence.scales,
            "native_norm": {k: v.flatten().tolist() for k, v in self.norm.items()},
            "sha256": {k: tensor_hash(v) for k, v in self.norm.items()},
        })
        development = [key for key in self.meta if self.meta[key]["split"] == "development"]
        self.cache, self.old_scores = self.evidence.load_development_paths(development)
        dump(self.destination / "SOURCE_HASHES.json", self.evidence.files)
        dump(self.destination / "ENVIRONMENT.json", dict(
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), cudnn=torch.backends.cudnn.version(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic, fp32_weights_and_inputs=True,
            tf32_override=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "unset"),
        ))
        self.precision = self.runtime_flags()
        self.weight_hashes = self.check_weights()
        dump(self.destination / "WEIGHT_HASHES.json", dict(before=self.weight_hashes))

    @staticmethod
    def runtime_flags():
        return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def check_weights(self):
        frozen = read(OUT / "FROZEN.json")
        training = read(OUT / "TRAINING_COMPLETE.json")
        assert freeze_hash(self.native, exclude_out=True) == frozen["frozen_non_out"]
        values = dict(non_out=freeze_hash(self.native, exclude_out=True))
        for name, out in self.outs.items():
            self.native.net.out.load_state_dict(out, strict=True)
            value = state_hash(self.native.net.out)
            expected = (training["backbone_out_initial_hash"] if name == "native"
                        else self.evidence.checkpoints[name]["out_state_hash"])
            assert value == expected
            values[f"{name}_out"] = value
        for arm, correction in self.corrections.items():
            values[f"{arm}_G"] = state_hash(correction)
            assert values[f"{arm}_G"] == self.evidence.checkpoints[arm]["correction_state_hash"]
        assert all(not p.requires_grad and p.dtype == torch.float32 and p.grad is None
                   for model in (self.native, *self.corrections.values()) for p in model.parameters())
        return values

    def predict(self, key, arm, state, reason, amplitudes=None, grad=False, ablation=None, inputs=None, trace=None):
        out = arm if state in ("O", "J") else "native"
        self.native.net.out.load_state_dict(self.outs[out], strict=True)
        assert state_hash(self.native.net.out) == self.weight_hashes[f"{out}_out"]
        data = self.inputs[key][arm] if inputs is None else inputs
        hashes = self.original_condition_hashes[key][arm]
        assert input_hashes(data) == hashes
        if ablation:
            data = dict(data)
            fields = ("E",) if ablation == "E_off" else ("W", "W_mean")
            for field in fields:
                data[field] = torch.zeros_like(data[field])
            actual = input_hashes(data)
            assert {f for f in data if actual[f] != hashes[f]} == set(fields)
        else:
            actual = hashes
        correction = self.corrections[arm] if state in ("G", "J") else None
        assert self.prior_groups + self.counts["complete_forward_groups"] < 192
        assert not grad or self.counts["gradient_graphs"] < 12
        assert not grad or (amplitudes is not None and amplitudes.requires_grad)
        self.counts["complete_forward_groups"] += 1
        self.counts["gradient_graphs"] += int(grad)
        row = dict(group=self.counts["complete_forward_groups"], **self.meta[key], arm=arm, state=state,
                   reason=reason, gradient_graph=grad, ablation=ablation or "",
                   amplitudes="" if amplitudes is None else str(amplitudes.detach().cpu().tolist()),
                   **{f"{f}_sha256": sha for f, sha in actual.items()}, output_sha256="")
        self.forward_log.append(row)
        assert self.runtime_flags() == self.precision
        with torch.set_grad_enabled(grad):
            torch.cuda.synchronize()
            result = sample(self.native, data, correction, amplitudes, trace)
            torch.cuda.synchronize()
        assert self.runtime_flags() == self.precision
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
        row["output_sha256"] = tensor_hash(result)
        return result

    def probes(self):
        key = next(iter(self.cache))
        result = []
        for arm, state, cache_arm in (("C0", "N", "native"), ("C0", "J", "C0"), ("C1", "J", "C1")):
            trace = []

            def record(label, x, features, delta, clean):
                trace.append(dict(label=label, x_sha256=tensor_hash(x), up2_sha256=tensor_hash(features),
                                  delta_sha256=tensor_hash(delta), clean_sha256=tensor_hash(clean),
                                  delta_rms=float(delta.double().square().mean().sqrt())))

            paths = self.predict(key, arm, state, "cache_reproduction", trace=record)
            cached = self.cache[key][cache_arm]
            maxabs = float((paths.cpu() - cached).abs().max())
            result.append(dict(cache_key=key, arm=cache_arm, maxabs=maxabs,
                               identical=maxabs == 0., sha256=tensor_hash(paths), stages=trace))
            dump(self.destination / "CACHE_REPRODUCTION.json", result)
            assert maxabs == 0., f"{cache_arm} does not reproduce the original cache, maxabs={maxabs}"
            if state == "J":
                assert tuple(t["label"] for t in trace) == LABELS
                assert all(t["delta_rms"] > 0 for t in trace)
        poisoned = self.predict(key, "C1", "J", "future_label_invariance", inputs=self.poisoned)
        assert torch.equal(poisoned, paths)
        dump(self.destination / "PUBLICATION_AUDIT.json", dict(
            cache_key=key, perturbed_target="reverse time axis then add 10 physical units",
            all_input_slots_identical=True, output_identical=True, output_sha256=tensor_hash(poisoned),
        ))

    def offline_decomposition(self):
        rows = []
        max_mse_error = 0.
        for key, cache in self.cache.items():
            native = cache["native"].cuda()
            for arm in ARMS:
                states = decompose(native, cache[arm].cuda())
                scores = {s: score_rows(self.meta[key], arm, s, p, self.targets[key], self.norm)
                          for s, p in states.items()}
                for p0, pshape in zip(scores["P0"], scores["P_shape"]):
                    for metric in ("mse", "smse"):
                        max_mse_error = max(max_mse_error, abs(p0[metric] - pshape[metric]))
                for state, old_arm, step in (("P0", "native", 0), ("P1", arm, 250)):
                    for row in scores[state]:
                        old = self.old_scores[old_arm, step, row["trajectory_id"], row["channel"]]
                        assert all(abs(row[m] - float(old[m])) < 1e-12 for m in METRICS)
                rows.extend(row for score_set in scores.values() for row in score_set)
        assert max_mse_error < 1e-12
        save_scores(self.destination, "DECOMPOSITION", rows, DECOMPOSITION)
        old_macro_errors = []
        with (OUT / "DEVELOPMENT_MACRO.csv").open() as stream:
            for row in csv.DictReader(stream):
                error = float(row["crps"]) - float(row["fair_crps"]) - float(row["pair_half"]) / 31
                old_macro_errors.append(abs(error))
        assert max(old_macro_errors) < 1e-12
        dump(self.destination / "DECOMPOSITION_CHECKS.json", dict(
            center_and_all_pairwise_differences_checked=True, both_mse_scales_maxabs=max_mse_error,
            cached_path_scores_match_old_rows=True, old_macro_crps_identity_maxabs=max(old_macro_errors),
        ))
        self.checkpoint("offline_decomposition")

    def components(self):
        self.core_scores, self.core_hashes = {}, {}
        for key in self.meta:
            has_cache = key in self.cache
            native = self.cache[key]["native"].cuda() if has_cache else self.predict(key, "C0", "N", "component")
            for arm in ARMS:
                joint = self.cache[key][arm].cuda() if has_cache else self.predict(key, arm, "J", "component")
                output = self.predict(key, arm, "O", "component")
                correction = self.predict(key, arm, "G", "component")
                for state, p in dict(N=native, J=joint, O=output, G=correction).items():
                    self.core_scores[key, arm, state] = score_rows(
                        self.meta[key], arm, state, p, self.targets[key], self.norm,
                    )
                    self.core_hashes[key, arm, state] = tensor_hash(p)
            # Flush complete windows so an abrupt resource stop still preserves results.
            csvwrite(self.destination / "COMPONENT_SCORES.csv", self.core_scores_flat())
        save_scores(self.destination, "COMPONENT", self.core_scores_flat(), COMPONENT)
        self.checkpoint("core_components")

    def core_scores_flat(self):
        return [dict(output_sha256=self.core_hashes[key], **row)
                for key, rows in self.core_scores.items() for row in rows]

    def amplitude(self):
        rows = []
        for key in self.meta:
            half = self.predict(key, "C1", "J", "half_amplitude", amplitudes=torch.full((4,), .5, device="cuda"))
            rows.extend(dict(output_sha256=tensor_hash(half), **r) for r in score_rows(
                self.meta[key], "C1", "half", half, self.targets[key], self.norm,
            ))
            for state in ("N", "O", "J"):
                core = (key, "C1", state)
                rows.extend(dict(output_sha256=self.core_hashes[core], **r) for r in self.core_scores[core])
        save_scores(self.destination, "AMPLITUDE", rows, AMPLITUDE)
        self.checkpoint("half_amplitude")

    def stages(self):
        gradients, base_grad = [], {}
        for key in self.stage:
            target = self.normalized_target(key)
            for endpoint, state in ((0, "O"), (1, "J")):
                parameters = torch.full((4,), float(endpoint), device="cuda", dtype=torch.float32, requires_grad=True)
                output = self.predict(key, "C1", "J", "stage_gradient", amplitudes=parameters, grad=True)
                output_hash = tensor_hash(output)
                assert output_hash == self.core_hashes[key, "C1", state]
                paths = output.double()
                for fair in (False, True):
                    metric = "fair_crps" if fair else "crps"
                    loss = crps(paths, target, fair)
                    assert self.counts["autograd_grad_calls"] < 24
                    self.counts["autograd_grad_calls"] += 1
                    self.gradient_log.append(dict(request=self.counts["autograd_grad_calls"], **self.meta[key],
                                                  endpoint=endpoint, metric=metric, output_sha256=output_hash))
                    (grad,) = torch.autograd.grad(loss, parameters, retain_graph=not fair)
                    assert torch.isfinite(grad).all()
                    values = grad.tolist()
                    base_grad[key, endpoint, metric] = values
                    gradients.append(dict(**self.meta[key], endpoint=endpoint, reference=state, metric=metric,
                                          loss=float(loss), **{f"label_{v}": g for v, g in zip(LABELS, values)},
                                          all_stages=sum(values), output_sha256=output_hash))
                del paths, output, loss
            csvwrite(self.destination / "STAGE_GRADIENTS.csv", gradients)
        self.checkpoint("stage_gradients")
        self.finite_differences(base_grad)

    def normalized_target(self, key):
        return (self.targets[key].double() - self.norm["mean"].double()) / self.norm["std"].double()

    def finite_differences(self, gradients):
        raw, assessed = [], []
        for key in self.fd_keys:
            directions = {"all": [1., 1., 1., 1.]}
            if key == self.fd_keys[0]:
                directions.update(label_99=[1., 0., 0., 0.], label_0=[0., 0., 0., 1.])
            target = self.normalized_target(key)
            for endpoint in (0, 1):
                for name, vector in directions.items():
                    slopes = {}
                    for h in (.01, .005):
                        values = {}
                        for sign in (-1, 1):
                            amplitude = torch.tensor([endpoint + sign*h*d for d in vector], device="cuda")
                            paths = self.predict(key, "C1", "J", f"fd_{name}", amplitudes=amplitude)
                            values[sign] = {m: float(crps(paths.double(), target, m == "fair_crps"))
                                            for m in ("crps", "fair_crps")}
                            raw.append(dict(**self.meta[key], endpoint=endpoint, direction=name, h=h, side=sign,
                                            **values[sign], output_sha256=tensor_hash(paths)))
                        slopes[h] = {m: (values[1][m] - values[-1][m]) / (2*h) for m in values[1]}
                    for metric in ("crps", "fair_crps"):
                        derivative = sum(d*g for d, g in zip(vector, gradients[key, endpoint, metric]))
                        first, second = slopes[.01][metric], slopes[.005][metric]
                        assessed.append(dict(**self.meta[key], endpoint=endpoint, direction=name, metric=metric,
                                             autograd=derivative, slope_h_0p01=first, slope_h_0p005=second,
                                             status=fd_status(derivative, first, second)))
                    csvwrite(self.destination / "FD_RUNS.csv", raw)
                    csvwrite(self.destination / "STAGE_FINITE_DIFFERENCES.csv", assessed)
        self.checkpoint("finite_differences")

    def conditions(self):
        rows, changes = [], []
        for key in self.stage:
            joint = self.cache[key]["C1"].cuda().double()
            rows.extend(self.core_scores[key, "C1", "J"])
            for state in ("E_off", "W_off"):
                paths = self.predict(key, "C1", "J", "condition_ablation", ablation=state)
                rows.extend(score_rows(self.meta[key], "C1", state, paths, self.targets[key], self.norm))
                diff = paths.double() - joint
                base_rms = float(joint.square().mean().sqrt())
                changes.append(dict(**self.meta[key], state=state,
                                    rms=float(diff.square().mean().sqrt()),
                                    mean_field_rms=float(diff.mean(0).square().mean().sqrt()),
                                    relative_to_J_rms=float(diff.square().mean().sqrt()) / base_rms,
                                    maxabs=float(diff.abs().max()), output_sha256=tensor_hash(paths)))
        csvwrite(self.destination / "CONDITION_OUTPUT_CHANGES.csv", changes)
        save_scores(self.destination, "CONDITION", rows, CONDITION)
        self.checkpoint("conditions")

    def run(self):
        self.prepare()
        self.probes()
        self.checkpoint("real_reproduction")
        self.offline_decomposition()
        self.components()
        self.amplitude()
        self.stages()
        self.conditions()
        final = self.check_weights()
        assert final == self.weight_hashes
        dump(self.destination / "WEIGHT_HASHES.json", dict(before=self.weight_hashes, after=final))
        for name, sha in self.evidence.files.items():
            self.evidence.check(ROOT / name, sha)
        assert all(input_hashes(inputs) == self.original_condition_hashes[key][arm]
                   for key, arms in self.inputs.items() for arm, inputs in arms.items())
        assert (self.counts["complete_forward_groups"], self.counts["gradient_graphs"],
                self.counts["autograd_grad_calls"]) == (176, 12, 24)
        dump(self.destination / "VERIFICATION.json", dict(
            passed=True, exact_cache_probes=3, future_label_invariance=True,
            all_publication_inputs_unchanged=True, weights_before_eq_after=True,
            all_checked_original_sources_and_artifacts_unchanged=True,
            member_independence="DynamicsNet/ResidualBlock/Correction have no operation reducing or mixing batch members",
            **self.counts,
        ))
        self.checkpoint("verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic-only", action="store_true")
    mode.add_argument("--cached-only", action="store_true",
                      help="score and decompose the locked historical cache without model inference")
    parser.add_argument("--legacy-precision", action="store_true",
                        help="match the archived CUDA matmul TF32 setting and require exact cache replay")
    parser.add_argument("--prior-cost", type=Path, help="include completed prior forwards in the hard cap")
    args = parser.parse_args()
    if args.legacy_precision and (args.synthetic_only or args.cached_only or not args.prior_cost):
        parser.error("--legacy-precision requires a full run and --prior-cost")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be empty")
    prior_groups = read(args.prior_cost)["complete_forward_groups"] if args.prior_cost else 0
    assert isinstance(prior_groups, int) and 0 <= prior_groups < 192
    diagnostic = Diagnostic(args.output, legacy_precision=args.legacy_precision, prior_groups=prior_groups)
    try:
        with TemporaryDirectory() as temporary:
            dump(args.output / "SYNTHETIC.json", run_synthetic(Path(temporary)))
        diagnostic.checkpoint("synthetic")
        if not args.synthetic_only:
            assert torch.cuda.is_available(), "CUDA runtime required"
            assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") == str(int(args.legacy_precision))
            if args.cached_only:
                diagnostic.prepare()
                diagnostic.offline_decomposition()
            else:
                diagnostic.run()
    except Exception as error:
        dump(args.output / "STOP.json", dict(error=repr(error), elapsed=perf_counter() - diagnostic.started))
        raise
    finally:
        diagnostic.save_cost()


if __name__ == "__main__":
    main()
