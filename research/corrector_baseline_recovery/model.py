"""Release adapters for the two architectures, using identical physical units."""
import torch

from research.corrector_consistency.model import PointCorrector

from .crossfit_corrector import BlockResidualCorrector


class BlockDelta(BlockResidualCorrector):
    """Adapt the original full-field return to the shared residual interface."""

    def forward(self, history, point):
        return super().forward(history, point) - point


def network(kind, device):
    if kind == "Point":
        return PointCorrector().to(device=device, memory_format=torch.channels_last)
    assert kind == "Block"
    model = BlockDelta().to(device)
    model.residual.to(memory_format=torch.channels_last_3d)
    return model


def input_fields(data, mean, units):
    return (data.x-mean)/units, (data.p-mean)/units


@torch.inference_mode()
def residual(model, fields, scale):
    model.eval()
    return torch.cat([model(*xs)*scale for xs in zip(*(x.split(8) for x in fields))])


def issue(point, predicted_residual, alpha):
    return point if alpha == 0. else point + alpha * predicted_residual
