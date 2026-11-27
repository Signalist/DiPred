"""Single-window publication with no previous-prediction argument."""
from statistics import median
from time import perf_counter

import torch

from research.corrector_residual_feedback.model import base


class Publisher:
    def __init__(self, backbone, s3, norm, s3_scale, residual_scale, head=None):
        self.backbone, self.s3, self.head = backbone, s3, head
        self.norm, self.s3_scale, self.residual_scale = norm, s3_scale, residual_scale
        for net in (backbone, s3, head):
            if net is not None:
                net.eval().requires_grad_(False)

    def native(self, observed):
        mean, std = self.norm["mean"], self.norm["std"]
        return self.backbone.rollout((observed-mean)/std, seed=500000, sign=0)*std+mean

    def correct(self, design):
        return self.s3(design)*self.s3_scale

    def increment(self, design, current):
        features = torch.cat((design, (current/self.residual_scale).float()), dim=1)
        return self.head(features).double()*self.residual_scale

    @torch.inference_mode()
    def __call__(self, observed):
        if observed.shape != (1, 2, 20, 32, 64) or observed.dtype != torch.float32:
            raise ValueError("Expected one FP32 observed window with shape [1, 2, 20, 32, 64].")
        native = self.native(observed)
        design = base(observed, native, self.norm)
        current = self.correct(design)
        correction = current.double()
        if self.head is not None:
            correction = correction + self.increment(design, current)
        return native.double() + correction


@torch.inference_mode()
def benchmark(publisher, observed, repeat=21, warmup=5):
    device = observed.device
    native = publisher.native(observed)
    design = base(observed, native, publisher.norm)
    current = publisher.correct(design)
    phases = dict(native=lambda:publisher.native(observed), s3=lambda:publisher.correct(design),
                  end_to_end=lambda:publisher(observed))
    if publisher.head is not None:
        phases["increment"] = lambda:publisher.increment(design, current)
    measurements = {}
    for name, fn in phases.items():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        initial = torch.cuda.memory_allocated(device)
        samples = []
        for _ in range(repeat):
            start = perf_counter()
            fn()
            torch.cuda.synchronize(device)
            samples.append(1000*(perf_counter()-start))
        measurements.update({f"ms_{name}_{k}":v for k,v in zip(("median","min","max"),
                             (median(samples),min(samples),max(samples)))})
        measurements[f"bytes_{name}_peak_delta"] = torch.cuda.max_memory_allocated(device)-initial
    head = publisher.head
    modules = (publisher.backbone, publisher.s3, head)
    total = sum(p.numel() for net in modules if net is not None for p in net.parameters())
    head_params = sum(p.numel() for p in head.parameters()) if head is not None else 0
    return dict(measurements,total_parameters=total,increment_parameters=head_params,
                repeat=repeat,warmup=warmup,batch=1,hardware=torch.cuda.get_device_name(device))


def load_publisher():
    """Load the frozen study candidate using packaged weights and signed metadata."""
    from reliability.backbone import configure
    from reliability.networks import DynamicsDiffusion20
    from research.corrector.run import sha
    from research.corrector_incremental_factorial.core import load_head
    from research.corrector_spacetime_validation.fit import load_net
    from research.corrector_window_study.data import ROOT, read_json

    from .run import OUT, STRUCTURE

    configure()
    frozen = read_json(OUT/"FROZEN.json")
    native_manifest = OUT/"native_development.json"
    assert sha(native_manifest) == frozen["bindings"][native_manifest.name]
    identity = read_json(native_manifest)["added_recipe"]["backbone"]
    native_path = ROOT/"reliability/checkpoints/current.pt"
    assert sha(native_path) == identity["packaged_checkpoint_sha256"]
    state = torch.load(native_path, map_location="cuda", weights_only=False)
    assert state["format_version"] == 1
    backbone = DynamicsDiffusion20(state["dynamics_scale"]).cuda()
    backbone.load_state_dict(state["predictor"], strict=True)
    source = frozen["source_s3"]
    device = next(backbone.parameters()).device
    teacher = load_net(ROOT/source["path"], source["sha256"], "S3", device, source["layout"])
    candidate = frozen["models"][frozen["selection"]["candidate"]]
    basis_path = STRUCTURE/"temporal_basis.pt"
    assert sha(basis_path)==frozen["basis_sha256"]
    basis = torch.load(basis_path, map_location="cpu", weights_only=True)
    head = load_head(candidate, candidate["final"], basis, device)
    s3_scale = frozen["models"]["S3_extra_MSE"]["config"]["scale"]
    return Publisher(backbone, teacher, state["normalization"], s3_scale, candidate["config"]["scale"], head)
