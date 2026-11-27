"""Frozen design entry point for the DiPred residual factorial study."""
import argparse
from pathlib import Path

from reliability.backbone import configure
from research.corrector_residual_feedback.data import ROOT
from research.corrector_spacetime_optimization.data import Study
from research.corrector_window_study.data import RAW

OUT = ROOT / "results/corrector_incremental_factorial"
CACHE = ROOT / ".cache/corrector_incremental_factorial"
FIGURES = ROOT / ".User/corrector_incremental_factorial"
PREVIOUS = ROOT / "results/corrector_incremental_structure"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", choices=("core", "all"), default="all")
    p.add_argument("--raw", type=Path, default=RAW)
    args = p.parse_args()
    configure()
    study = Study(OUT, CACHE, args.raw)
    if not (OUT / "FROZEN.json").exists():
        from .core import core

        core(study)
    if args.action == "all":
        from .validation import validate

        validate(study, FIGURES)


if __name__ == "__main__":
    main()
