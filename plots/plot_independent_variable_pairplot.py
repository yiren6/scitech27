#!/usr/bin/env python3
"""Pair plot of independent variables for training and validation geometries.

Loads MF pickle cases from the training-geometry and validation-geometry
directories, extracts (Re, M, α, NACA camber X%, NACA thickness ZE%, Λ),
and draws a lower-triangle scatter pair plot with per-split KDEs on the
diagonal. The upper triangle is left empty.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOTS_DIR.parent
DEFAULT_TRAIN_DIR = Path("/Users/yirenshen/Desktop/MF-VortexNet/dataset/train_set")
DEFAULT_TEST_DIR = Path("/Users/yirenshen/Desktop/MF-VortexNet/dataset/new_geom")
DEFAULT_OUTPUT = PLOTS_DIR / "independent_variable_pairplot.png"

FILENAME_RE = re.compile(r"_sweep_(\d+)_naca_(\d+)", re.IGNORECASE)

VARS = ["Re", "M", "alpha", "camber", "thickness", "sweep"]
VAR_LABELS = {
    "Re": r"$Re$",
    "M": r"$M$",
    "alpha": r"$\alpha$ (deg)",
    "camber": r"$X$ (%)",
    "thickness": r"$ZE$ (%)",
    "sweep": r"$\Lambda$ (deg)",
}


class _Stub:
    """Stand-in for pickled SUAVE objects not needed for freestream/geometry metadata."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]


_ALLOWED_MODULES = (
    "MFData",
    "VortexNet",
    "numpy",
    "collections",
    "copyreg",
    "builtins",
    "__builtin__",
    "scipy",
)


class _NoSuaveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_ALLOWED_MODULES) or module in _ALLOWED_MODULES:
            return super().find_class(module, name)

        parts = module.split(".")
        for index in range(1, len(parts) + 1):
            module_name = ".".join(parts[:index])
            if module_name not in sys.modules:
                imported_module = types.ModuleType(module_name)
                sys.modules[module_name] = imported_module
                if index > 1:
                    setattr(
                        sys.modules[".".join(parts[: index - 1])],
                        parts[index - 1],
                        imported_module,
                    )
        imported_module = sys.modules[module]
        if not hasattr(imported_module, name):
            setattr(
                imported_module,
                name,
                type(name, (_Stub,), {"__module__": module}),
            )
        return getattr(imported_module, name)


def _register_mfdata() -> None:
    import importlib.util

    mfdata_path = REPO_ROOT / "VortexNet" / "MFData.py"
    spec = importlib.util.spec_from_file_location("MFData", mfdata_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MFData from {mfdata_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["MFData"] = module
    spec.loader.exec_module(module)


def _parse_geometry(filename: str) -> tuple[float, float, float]:
    match = FILENAME_RE.search(filename)
    if match is None:
        raise ValueError(f"Could not parse sweep/NACA from filename: {filename}")
    sweep = float(match.group(1))
    naca = match.group(2).zfill(4)
    camber = float(int(naca[0]))  # X% in NACAXYZE
    thickness = float(int(naca[2:]))  # ZE% in NACAXYZE
    return sweep, camber, thickness


def _load_cases(directory: Path, *, validation: bool) -> list:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    files = sorted(directory.glob("pklData_*.pkl"))
    if validation:
        files = [path for path in files if "validation" in path.name]
    else:
        files = [path for path in files if "validation" not in path.name]
    if not files:
        split_name = "validation" if validation else "training"
        raise FileNotFoundError(f"No {split_name} pklData files found in {directory}.")

    cases = []
    for path in files:
        with path.open("rb") as handle:
            loaded = _NoSuaveUnpickler(handle).load()
        if not isinstance(loaded, list):
            raise TypeError(f"{path.name} did not contain a list of cases.")
        for case in loaded:
            case.source_file = path.name
        cases.extend(loaded)
        print(f"Loaded {len(loaded):4d} cases from {path.name}")
    print(f"Total cases from {directory}: {len(cases)}")
    return cases


def cases_to_dataframe(cases: list, split: str) -> pd.DataFrame:
    rows = []
    for case in cases:
        sweep, camber, thickness = _parse_geometry(case.source_file)
        rows.append(
            {
                "split": split,
                "Re": float(case.Re),
                "M": float(case.mach),
                "alpha": float(case.alpha),
                "camber": camber,
                "thickness": thickness,
                "sweep": sweep,
            }
        )
    return pd.DataFrame(rows)


def build_pairplot(df: pd.DataFrame, output_path: Path, font_size: int = 20) -> None:
    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
        }
    )

    hue_order = ["Training", "Validation"]
    palette = {"Training": "#1f77b4", "Validation": "#d62728"}
    markers = {"Training": "o", "Validation": "^"}

    plot_df = df.copy()
    # Compact Re axis for readability while keeping the Re label.
    plot_df["Re"] = plot_df["Re"] / 1.0e6

    g = sns.PairGrid(
        plot_df,
        vars=VARS,
        hue="split",
        hue_order=hue_order,
        palette=palette,
        hue_kws={"marker": [markers[name] for name in hue_order]},
        diag_sharey=False,
        corner=True,  # leave the upper triangle empty
        height=2.6,
        aspect=1.0,
    )

    g.map_lower(
        plt.scatter,
        s=40,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
    )
    g.map_diag(
        sns.kdeplot,
        common_norm=False,
        fill=True,
        alpha=0.35,
        linewidth=2.0,
        warn_singular=False,
    )

    for i, y_var in enumerate(VARS):
        for j, x_var in enumerate(VARS):
            ax = g.axes[i, j]
            if ax is None:
                continue
            if i == len(VARS) - 1:
                ax.set_xlabel(VAR_LABELS[x_var], fontsize=font_size)
            else:
                ax.set_xlabel("")
            if j == 0:
                ylabel = VAR_LABELS[y_var]
                if y_var == "Re":
                    ylabel = r"$Re$ ($10^{6}$)"
                ax.set_ylabel(ylabel, fontsize=font_size)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=font_size)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(font_size)

    # Single legend outside the lower-left panel.
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[name],
            color="none",
            markerfacecolor=palette[name],
            markeredgecolor="white",
            markersize=12 if name == "Training" else 13,
            label=name,
        )
        for name in hue_order
    ]
    g.fig.legend(
        handles=handles,
        labels=hue_order,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=font_size,
        title="Geometry set",
        title_fontsize=font_size,
    )

    g.fig.subplots_adjust(top=0.96, right=0.96, wspace=0.12, hspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved pair plot to {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-size", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _register_mfdata()

    train_cases = _load_cases(args.train_dir.expanduser().resolve(), validation=False)
    test_cases = _load_cases(args.test_dir.expanduser().resolve(), validation=True)

    df = pd.concat(
        [
            cases_to_dataframe(train_cases, "Training"),
            cases_to_dataframe(test_cases, "Validation"),
        ],
        ignore_index=True,
    )
    print(
        f"Pair-plot rows: {len(df)} "
        f"(Training={int((df['split'] == 'Training').sum())}, "
        f"Validation={int((df['split'] == 'Validation').sum())})"
    )
    print(df.groupby("split")[VARS].agg(["min", "max"]).to_string())

    build_pairplot(df, args.output.expanduser().resolve(), font_size=args.font_size)


if __name__ == "__main__":
    main()
