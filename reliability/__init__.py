"""Native four-call DiPred with direct predictive-error reliability."""

__all__ = ["ReliabilityPredictor", "MemoryReliabilityPredictor", "ContinuedReliabilityPredictor"]


def __getattr__(name):
    # Keep `python -m reliability.predict` free of duplicate module loading.
    if name == "ReliabilityPredictor":
        from .predict import ReliabilityPredictor
        return ReliabilityPredictor
    if name == "MemoryReliabilityPredictor":
        from .memory_predict import MemoryReliabilityPredictor
        return MemoryReliabilityPredictor
    if name == "ContinuedReliabilityPredictor":
        from .continued_predict import ContinuedReliabilityPredictor
        return ContinuedReliabilityPredictor
    raise AttributeError(name)
