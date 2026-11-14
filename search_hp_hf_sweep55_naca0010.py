"""Optuna search for the HF autoregressor trained on 55° sweep NACA 0010 only.

The study still minimizes holdout RMSE on the random 20% of that wing.
dataset/new_geom is scored and saved for every trial, and it does not choose
the hyperparameters. Default mode is a short test run.

    python VortexNet/hf_sweep55_naca0010/search_hp_hf_sweep55_naca0010.py
    python VortexNet/hf_sweep55_naca0010/search_hp_hf_sweep55_naca0010.py --mode full
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from search_hp_hf_autoregressor import (  # noqa: E402
    parse_args,
    search,
)

TRAIN_FILE_CONTAINS = "sweep_55_naca_0010"


def main() -> None:
    args = parse_args()
    args.train_file_contains = TRAIN_FILE_CONTAINS
    if args.output_dir is None:
        folder = "optuna_results_550010" if args.mode == "full" else "optuna_results_test_550010"
        args.output_dir = str(PACKAGE_DIR / folder)
    if args.mode == "full":
        args.study_name = "hf_sweep55_naca0010_7030"
    else:
        args.study_name = "hf_sweep55_naca0010_test"
    search(args)


if __name__ == "__main__":
    main()
