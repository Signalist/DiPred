"""Smooth revealed overall interval scales, leaving risk shape unchanged."""

from moment_online_risk import MomentRiskCalibrator, controller as reference_controller


MODES = (
    "moment_dynamic", "moment_fastcal", "quantile_dynamic",
    "scale_dynamic", "scale_fastcal",
)


class SmoothedScaleCalibrator(MomentRiskCalibrator):
    """Use a log-scale EMA after the first observed trajectory window.

    The first feedback sets a trajectory-specific amplitude directly. Later
    amplitudes use the geometric mean of the previous and newly fitted scales.
    All fitting still follows the full 20-frame feedback, never a partial block.
    """

    def __init__(self, mode, initial, batch):
        base_mode = "moment_fastcal" if mode == "scale_fastcal" else "moment_dynamic"
        super().__init__(base_mode, initial, batch)

    def reveal(self, residual):
        old_amplitude = self.amplitude.clone()
        completed = self.revealed_windows
        super().reveal(residual)
        if completed > 0:
            self.amplitude = (old_amplitude * self.amplitude).sqrt()


def controller(mode, initial, batch):
    if mode.startswith("scale_"):
        return SmoothedScaleCalibrator(mode, initial, batch)
    return reference_controller(mode, initial, batch)
