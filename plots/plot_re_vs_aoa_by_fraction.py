#!/usr/bin/env python3
"""Re vs AoA plots of training snapshots and the full test geometry set.

For each scaling experiment (HF, MF fixed-HP, MF autohp), each run, and each
training-set fraction, this script:

1. Reads the recorded train-set split CSV to recover which training-geometry
   cases belong to that nested snapshot (fit + holdout rows).
2. Overlays the entire lhs-free-stream generalized-geometry test set.
3. Writes one Re–AoA figure (font size 20).

Outputs land under ``plots/re_vs_aoa/<experiment>/``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent
REPO = PLOTS_DIR.parent
DEFAULT_TEST_DIR = Path(
    "/home/yiren/Desktop/MF-VortexNet/dataset/new_geom/lhs-free-stream"
)
FONT = 20

# Sorted seed order matches report_all_scaling.py Run 1..5 mapping.
SEEDS = (7, 27, 207, 353, 3407)
RUN_OF_SEED = {seed: index for index, seed in enumerate(SEEDS, start=1)}

EXPERIMENTS = (
    {
        "key": "train_hf_scaling",
        "title": "HF autoregressor",
        "results_root": REPO / "train_hf_scaling_results",
        "kind": "hf",
    },
    {
        "key": "train_mf_scaling",
        "title": "MF fixed HP",
        "results_root": REPO / "train_mf_scaling_results",
        "kind": "mf",
    },
    {
        "key": "train_mf_scaling_autohp",
        "title": "MF + autohp",
        "results_root": REPO / "train_mf_scaling_autohp_results",
        "kind": "mf",
    },
)

TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#d62728"


class _Stub:
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


_ALLOWED = (
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
        if module.startswith(_ALLOWED) or module in _ALLOWED:
            return super().find_class(module, name)
        parts = module.split(".")
        for index in range(1, len(parts) + 1):
            module_name = ".".join(parts[:index])
            if module_name not in sys.modules:
                mod = types.ModuleType(module_name)
                sys.modules[module_name] = mod
                if index > 1:
                    setattr(
                        sys.modules[".".join(parts[: index - 1])],
                        parts[index - 1],
                        mod,
                    )
        mod = sys.modules[module]
        if not hasattr(mod, name):
            setattr(mod, name, type(name, (_Stub,), {"__module__": module}))
        return getattr(mod, name)


def _register_mfdata() -> None:
    import importlib.util

    mfdata_path = REPO / "VortexNet" / "MFData.py"
    spec = importlib.util.spec_from_file_location("MFData", mfdata_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MFData from {mfdata_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["MFData"] = module
    spec.loader.exec_module(module)


def load_test_geometry(test_dir: Path) -> pd.DataFrame:
    """Load the full lhs-free-stream (or similar) test set freestreams."""
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)
    files = sorted(test_dir.glob("pklData_*.pkl"))
    preferred = [
        path
        for path in files
        if "validation" in path.name or "vortexNetNewGeom" in path.name
    ]
    files = preferred or files
    if not files:
        raise FileNotFoundError(f"No pklData_*.pkl in {test_dir}")

    rows = []
    for path in files:
        with path.open("rb") as handle:
            loaded = _NoSuaveUnpickler(handle).load()
        if not isinstance(loaded, list):
            raise TypeError(f"{path.name} did not contain a list")
        for case in loaded:
            rows.append(
                {
                    "mach": float(case.mach),
                    "alpha": float(case.alpha),
                    "Re": float(case.Re),
                    "source_file": path.name,
                }
            )
        print(f"Loaded {len(loaded):4d} test cases from {path.name}")
    df = pd.DataFrame(rows)
    print(f"Total test-geometry cases: {len(df)}")
    return df


def load_hf_splits(results_root: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = (
            results_root
            / f"train_hf_scaling_results_seed{seed}"
            / "train_set_split.csv"
        )
        if not path.is_file():
            print(f"Missing HF split CSV: {path}")
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["run"] = RUN_OF_SEED[seed]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No HF train_set_split.csv under {results_root}")
    return pd.concat(frames, ignore_index=True)


def load_mf_splits(results_root: Path) -> pd.DataFrame:
    path = results_root / "train_validation_split.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["run"] = frame["seed"].map(RUN_OF_SEED)
    missing = frame["run"].isna()
    if missing.any():
        unknown = sorted(frame.loc[missing, "seed"].unique())
        raise ValueError(f"Unexpected seeds in {path}: {unknown}")
    frame["run"] = frame["run"].astype(int)
    return frame


def snapshot_cases(split_df: pd.DataFrame, *, seed: int, fraction: float) -> pd.DataFrame:
    """All training-geometry cases belonging to the nested snapshot.

    HF uses split labels ``training`` / ``test`` (holdout).
    MF uses ``training`` / ``validation``. Both rows are part of the selected
    fraction subset.
    """
    subset = split_df[
        (split_df["seed"] == seed)
        & np.isclose(split_df["fraction"].astype(float), float(fraction))
    ].copy()
    if subset.empty:
        raise ValueError(f"No split rows for seed={seed}, fraction={fraction}")
    return subset


MARKER_ALPHA = 0.80
ALPHA_LIM = (-1.0, 21.0)
RE_LIM_1E6 = (6.0, 10.5)


def _style_re_axis(ax, font_size: int = FONT) -> None:
    """Non-italic 'Re' label with a \\times10^6 annotation on the y-axis."""
    ax.set_ylabel("Re", fontstyle="normal")
    ax.set_ylim(*RE_LIM_1E6)
    ax.text(
        0.0,
        1.01,
        r"$\times10^{6}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=font_size,
    )


def apply_style(font_size: int = FONT) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "figure.dpi": 120,
            "savefig.dpi": 200,
        }
    )


def plot_re_vs_aoa(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
    title: str,
    font_size: int = FONT,
    show_legend: bool = False,
) -> None:
    apply_style(font_size)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    ax.scatter(
        train_df["alpha"].to_numpy(dtype=float),
        train_df["Re"].to_numpy(dtype=float) / 1.0e6,
        s=55,
        alpha=MARKER_ALPHA,
        marker="o",
        color=TRAIN_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Train",
        zorder=2,
    )
    ax.scatter(
        test_df["alpha"].to_numpy(dtype=float),
        test_df["Re"].to_numpy(dtype=float) / 1.0e6,
        s=70,
        alpha=MARKER_ALPHA,
        marker="^",
        color=TEST_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Test",
        zorder=3,
    )

    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_xlim(*ALPHA_LIM)
    _style_re_axis(ax, font_size=font_size)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_re_vs_aoa_all_runs(
    *,
    train_by_run: dict[int, pd.DataFrame],
    test_df: pd.DataFrame,
    output_path: Path,
    title: str,
    font_size: int = FONT,
    show_legend: bool = False,
) -> None:
    """Optional overlay: all runs at one fraction + full test set."""
    apply_style(font_size)
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    run_colors = {
        1: "#1f4e79",
        2: "#d35400",
        3: "#7d3c98",
        4: "#566573",
        5: "#2e8b57",
    }
    run_markers = {1: "o", 2: "s", 3: "^", 4: "v", 5: "D"}
    for run, frame in sorted(train_by_run.items()):
        ax.scatter(
            frame["alpha"].to_numpy(dtype=float),
            frame["Re"].to_numpy(dtype=float) / 1.0e6,
            s=45,
            alpha=MARKER_ALPHA,
            marker=run_markers.get(run, "o"),
            color=run_colors.get(run, "#333333"),
            edgecolors="white",
            linewidths=0.3,
            label=f"Run {run}",
            zorder=2,
        )
    ax.scatter(
        test_df["alpha"].to_numpy(dtype=float),
        test_df["Re"].to_numpy(dtype=float) / 1.0e6,
        s=80,
        alpha=MARKER_ALPHA,
        marker="*",
        color=TEST_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Test",
        zorder=4,
    )
    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_xlim(*ALPHA_LIM)
    _style_re_axis(ax, font_size=font_size)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_experiment(
    experiment: dict,
    test_df: pd.DataFrame,
    output_root: Path,
    font_size: int,
) -> list[Path]:
    results_root = experiment["results_root"]
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)

    if experiment["kind"] == "hf":
        split_df = load_hf_splits(results_root)
    else:
        split_df = load_mf_splits(results_root)

    out_dir = output_root / experiment["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fractions = sorted(float(value) for value in split_df["fraction"].unique())
    seeds_present = sorted(int(value) for value in split_df["seed"].unique())

    for fraction in fractions:
        pct = int(round(fraction * 100))
        train_by_run: dict[int, pd.DataFrame] = {}
        for seed in seeds_present:
            run = RUN_OF_SEED[seed]
            snapshot = snapshot_cases(split_df, seed=seed, fraction=fraction)
            train_by_run[run] = snapshot
            show_legend = pct == 6
            path = out_dir / f"run{run}_fraction_{pct:02d}_re_vs_aoa.png"
            plot_re_vs_aoa(
                train_df=snapshot,
                test_df=test_df,
                output_path=path,
                title=f"Run {run}, {pct}% Training",
                font_size=font_size,
                show_legend=show_legend,
            )
            written.append(path)
            print(f"Wrote {path}")

        overlay = out_dir / f"all_runs_fraction_{pct:02d}_re_vs_aoa.png"
        plot_re_vs_aoa_all_runs(
            train_by_run=train_by_run,
            test_df=test_df,
            output_path=overlay,
            title=f"All runs, {pct}% Training",
            font_size=font_size,
            show_legend=show_legend,
        )
        written.append(overlay)
        print(f"Wrote {overlay}")

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR / "re_vs_aoa",
    )
    parser.add_argument("--font-size", type=int, default=FONT)
    parser.add_argument(
        "--which",
        nargs="+",
        choices=[item["key"] for item in EXPERIMENTS] + ["all"],
        default=["all"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _register_mfdata()
    test_df = load_test_geometry(args.test_dir.expanduser().resolve())

    which = set(args.which)
    if "all" in which:
        selected = list(EXPERIMENTS)
    else:
        selected = [item for item in EXPERIMENTS if item["key"] in which]

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_written: list[Path] = []
    for experiment in selected:
        print(f"\n=== {experiment['key']} ===")
        all_written.extend(
            process_experiment(
                experiment,
                test_df,
                output_root,
                font_size=args.font_size,
            )
        )

    print(f"\nWrote {len(all_written)} figures under {output_root}")


if __name__ == "__main__":
    main()
