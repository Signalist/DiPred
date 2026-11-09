"""Native four-call DiPred with direct predictive-error reliability."""

__all__ = ["ReliabilityPredictor"]


def __getattr__(name):
    # Keep `python -m reliability.predict` free of duplicate module loading.
    if name == "ReliabilityPredictor":
        from .predict import ReliabilityPredictor
        return ReliabilityPredictor
    raise AttributeError(name)
