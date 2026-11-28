"""Train the preregistered conditional-history contrasts and score frozen models."""
from argparse import ArgumentParser
from pathlib import Path

from reliability.backbone import configure
from research.corrector_spacetime_optimization.data import Study
from research.corrector_window_study.data import RAW, ROOT

NAME = "corrector_conditional_history"
OUT = ROOT / "results" / NAME
CACHE = ROOT / ".cache" / NAME
FIGURES = ROOT / ".User" / NAME
PREVIOUS = ROOT / "results/corrector_remaining_residual_transfer"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("core", "all"), nargs="?", default="all")
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    configure()
    study = Study(OUT, CACHE, args.raw)
    if not (OUT / "MAIN_FROZEN.json").exists():
        from .core import fit_core

        fit_core(study)
    if args.action == "all":
        from .evaluate import validate

        validate(study, FIGURES)


if __name__ == "__main__":
    main()
