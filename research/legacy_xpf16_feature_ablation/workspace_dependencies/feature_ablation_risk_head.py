"""Source-group ablations of the 28-channel dynamic quantile risk core.

The four historical core channels contain both projected physical history and
transformed revealed-error memory. Removing that group removes both sources.
All masks act after normalization/projection and on every modifier input path.
No future target is accepted by bounds(). All 20 leads and all grid cells remain.
"""
import math
import torch
import torch.nn.functional as F

from dynamic_risk_memory import DynamicRiskMemoryHead, warp_risk_features

BASE_GROUPS = {
    'forecast': (0, 1),
    'last_observation': (2, 3),
    'history_variability': (4, 5),
    'forecast_departure': (6, 7),
    'forecast_gradient': (8, 9),
    'forecast_time_change': (10, 11),
    'curl': (12,),
    'lead': (13,),
}
GROUPS = dict(BASE_GROUPS, history_memory=(14, 15, 16, 17),
              denoising=tuple(range(18, 26)), coordinates=(26, 27))
ARMS = ('full',) + tuple('drop_' + name for name in GROUPS)


class AblatedRiskHead(DynamicRiskMemoryHead):
    """Keep allocated parameters identical to isolate conditional input utility."""

    def __init__(self, memory_scale, original_state, dynamic_state, dropped=None):
        super().__init__(memory_scale, original_state, 'warp')
        if dropped is not None and dropped not in GROUPS:
            raise ValueError(f'Unknown feature group {dropped}.')
        self.load_state_dict(dynamic_state, strict=True)
        self.dropped = dropped
        core_mask = torch.ones(28)
        if dropped is not None:
            core_mask[list(GROUPS[dropped])] = 0
        # This is protocol metadata, not an independently fitted state tensor.
        self.register_buffer('core_mask', core_mask.reshape(1, 28, 1, 1, 1), persistent=False)

    def bounds(self, base, last, history, previous_error, available):
        batch, _, _, height, width = base.shape
        historical = self.history_projection(history.reshape(batch, 40, height, width))
        historical = historical.reshape(batch, 4, 20, height, width)
        encoded = self.encode_memory(previous_error).reshape(batch, 40, height, width)
        gate = torch.as_tensor(available, dtype=base.dtype, device=base.device)
        gate = gate.expand(batch) if gate.numel() == 1 else gate.reshape(batch)
        memory = self.memory_projection(encoded) * gate[:, None, None, None]
        memory = memory.reshape(batch, 4, 20, height, width)
        features = self.feature_projection(last).reshape(batch, 8, 20, height, width)
        yy = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
        xx = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
        y, x = torch.meshgrid(yy, xx, indexing='ij')
        coordinates = torch.stack([x, y])[None, :, None].expand(batch, 2, 20, height, width)
        # Mask after projection: a learned bias cannot reintroduce an omitted
        # source. Mask the modifier too, avoiding a route through displacement.
        base = base * self.core_mask[:, :14]
        historical = historical * self.core_mask[:, 14:18]
        memory = memory * self.core_mask[:, 14:18]
        features = features * self.core_mask[:, 18:26]
        coordinates = coordinates * self.core_mask[:, 26:28]
        raw = self.modifier(torch.cat([base, historical, features, coordinates, memory], dim=1))
        transformed = warp_risk_features(memory, 4 * raw.tanh())
        # Match the original flatten/add/reshape ordering bit-for-bit.
        historical = (historical.reshape(batch, 80, height, width)
                      + transformed.reshape(batch, 80, height, width)).reshape(batch, 4, 20, height, width)
        value = self.net(torch.cat([base, historical, features, coordinates], dim=1))
        midpoint = value[:, :2]
        radius = F.softplus(value[:, 2:]) + 1e-4
        return midpoint - radius, midpoint + radius
