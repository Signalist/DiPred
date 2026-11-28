"""Train, freeze and evaluate fixed-reference residual transfer."""
from argparse import ArgumentParser
from pathlib import Path

from reliability.backbone import configure
from research.corrector_spacetime_optimization.data import Study
from research.corrector_window_study.data import RAW, ROOT

NAME = "corrector_remaining_residual_transfer"
OUT = ROOT / "results" / NAME
CACHE = ROOT / ".cache" / NAME
FIGURES = ROOT / ".User" / NAME
REFINEMENT = ROOT / "results/corrector_history_free_refinement"
FACTORIAL = ROOT / "results/corrector_incremental_factorial"
STRUCTURE = ROOT / "results/corrector_incremental_structure"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("core", "all"), nargs="?", default="all")
    parser.add_argument("--raw", type=Path, default=RAW)
    options = parser.parse_args()
    configure()
    study = Study(OUT, CACHE, options.raw)
    if not (OUT / "FROZEN.json").exists():
        from .core import fit_core

        fit_core(study)
    if options.action == "all":
        from .evaluate import validate

        validate(study, FIGURES)


if __name__ == "__main__":
    main()
