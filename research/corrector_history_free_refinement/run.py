"""Run the history-free DiPred refinement study."""
import argparse
from pathlib import Path

from reliability.backbone import configure
from research.corrector_window_study.data import RAW, ROOT
from research.corrector_spacetime_optimization.data import Study

NAME = "corrector_history_free_refinement"
OUT = ROOT / "results" / NAME
CACHE = ROOT / ".cache" / NAME
FIGURES = ROOT / ".User" / NAME
FACTORIAL = ROOT / "results/corrector_incremental_factorial"
STRUCTURE = ROOT / "results/corrector_incremental_structure"
S3_RESULTS = ROOT / "results/corrector_residual_optimization"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("core", "all"), default="all")
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    configure()
    study = Study(OUT, CACHE, args.raw)
    if not (OUT / "FROZEN.json").exists():
        from .core import fit_main

        fit_main(study)
    if args.action == "all":
        from .evaluate import validate

        validate(study, FIGURES)


if __name__ == "__main__":
    main()
