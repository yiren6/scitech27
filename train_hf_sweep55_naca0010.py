"""Train the HF autoregressor on the 55° sweep NACA 0010 wing only.

The fit/holdout split is the random 80/20 of that one geometry. The sweep 60/70
wings in dataset/new_geom are still scored after training and are not used to
fit the model. Outputs stay in this folder.

    python VortexNet/hf_sweep55_naca0010/train_hf_sweep55_naca0010.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_hf_autoregressor import (  # noqa: E402
    PACKAGE_DIR as FULL_PACKAGE,
)
from train_hf_autoregressor import parse_args, train  # noqa: E402

TRAIN_FILE_CONTAINS = "sweep_55_naca_0010"


def main() -> None:
    args = parse_args()
    args.train_file_contains = TRAIN_FILE_CONTAINS
    if Path(args.output_dir) == FULL_PACKAGE / "output":
        args.output_dir = str(PACKAGE_DIR / "output")
    train(args)


if __name__ == "__main__":
    main()
