"""Write durable scaling reports and figures for HF / MF / MF-autohp.

Reads per-run ``report.json`` (or HF ``best/report.json``) that were rescored
on lhs-free-stream, and writes:

- ``train_hf_scaling_report/``
- ``train_mf_scaling_report/``
- ``train_mf_scaling_autohp_report/``

Each folder gets ``report.md``, ``seed_average.csv``, ``per_seed.json``, and
``figures/test_error_vs_training_snapshots.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
TEST_SET = (
    "`MF-VortexNet/dataset/new_geom/lhs-free-stream` "
    "(`*_vortexNetNewGeom.pkl`)"
)

METRIC_SPECS = (
    ("test_rmse", "Test RMSE (ΔCp)", "Unseen-geometry RMSE"),
    ("test_mae", "Test MAE (ΔCp)", "Unseen-geometry MAE"),
    ("test_nrmse_std", "Test NRMSE (std)", "Unseen-geometry NRMSE"),
)

# Stable mapping: sorted seed order -> Run 1..N (legend never shows raw seeds).
SEED_COLORS = {
    7: "#1f4e79",
    27: "#d35400",
    207: "#7d3c98",
    353: "#566573",
    3407: "#2e8b57",
}
SEED_MARKERS = {7: "o", 27: "s", 207: "^", 353: "v", 3407: "D"}
RUN_LINEWIDTH = 1.6
MEAN_LINEWIDTH = RUN_LINEWIDTH * 1.5
RUN_ALPHA = 0.5
FIGURE_NAMES = {
    "test_rmse": "test_rmse_vs_training_snapshots.png",
    "test_mae": "test_mae_vs_training_snapshots.png",
    "test_nrmse_std": "test_nrmse_std_vs_training_snapshots.png",
}


def run_labels(seeds: list[int]) -> dict[int, str]:
    return {seed: f"Run {index}" for index, seed in enumerate(seeds, start=1)}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )


def load_mf_runs(results_dir: Path) -> pd.DataFrame:
    records = []
    for report_path in sorted(results_dir.glob("seed_*/fraction_*/run/report.json")):
        report = json.loads(report_path.read_text())
        test = report["final_test_geometry"]
        validation = report["validation_geometry"]
        if not np.isclose(float(report["test_rmse"]), float(test["rmse"])):
            raise ValueError(f"Inconsistent test RMSE in {report_path}")
        records.append(
            {
                "seed": int(report["seed"]),
                "fraction": float(report["fraction"]),
                "n_selected": int(report["n_selected"]),
                "n_training": int(report["n_training"]),
                "n_validation": int(report["n_validation"]),
                "n_test_cases": int(report["n_test_cases"]),
                "validation_rmse": float(validation["rmse"]),
                "test_rmse": float(test["rmse"]),
                "test_mae": float(test["mae"]),
                "test_nrmse_std": float(test["nrmse_std"]),
                "best_trial": int(report["best_trial"])
                if "best_trial" in report
                else -1,
                "report_path": str(report_path),
            }
        )
    if not records:
        raise FileNotFoundError(f"No MF reports under {results_dir}")
    runs = pd.DataFrame.from_records(records).sort_values(["fraction", "seed"])
    return runs.reset_index(drop=True)


def load_hf_runs(results_root: Path) -> pd.DataFrame:
    records = []
    for seed_dir in sorted(results_root.glob("train_hf_scaling_results_seed*")):
        seed = int(seed_dir.name.split("seed")[-1])
        for report_path in sorted(seed_dir.glob("fraction_*/best/report.json")):
            report = json.loads(report_path.read_text())
            test = report["test_report"]
            holdout = report["training_geometry_holdout"]
            split = report["split"]
            if "training_fraction" in report:
                fraction = float(report["training_fraction"])
            else:
                fraction = int(report_path.parent.parent.name.split("_")[1]) / 100.0
            summary = seed_dir / "scaling_results.json"
            best_trial = -1
            if summary.is_file():
                for item in json.loads(summary.read_text()).get("results", []):
                    if round(float(item["fraction"]) * 100) == round(fraction * 100):
                        best_trial = int(item["best_trial"])
                        break
            records.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "n_selected": int(split["n_fit"]) + int(split["n_holdout"]),
                    "n_training": int(split["n_fit"]),
                    "n_validation": int(split["n_holdout"]),
                    "n_test_cases": int(split.get("n_test_geometry", test.get("n_cases", 0))),
                    "validation_rmse": float(holdout["rmse"]),
                    "test_rmse": float(test["rmse"]),
                    "test_mae": float(test["mae"]),
                    "test_nrmse_std": float(test["nrmse_std"]),
                    "best_trial": best_trial,
                    "report_path": str(report_path),
                }
            )
    if not records:
        raise FileNotFoundError(f"No HF best reports under {results_root}")
    return pd.DataFrame.from_records(records).sort_values(["fraction", "seed"]).reset_index(
        drop=True
    )


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs.groupby("fraction", as_index=False)
        .agg(
            n_selected=("n_selected", "first"),
            n_training=("n_training", "first"),
            n_validation=("n_validation", "first"),
            n_seeds=("seed", "nunique"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric, _, _ in METRIC_SPECS
                for stat in ("mean", "std")
            },
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
        )
        .sort_values("fraction")
        .reset_index(drop=True)
    )
    return summary


def plot_scaling(runs: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Write one log-log figure per metric. Returns the saved paths."""
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = summary["n_selected"].to_numpy(dtype=float)
    tick_labels = [f"{fraction:.0%}" for fraction in summary["fraction"]]
    seeds = sorted(int(s) for s in runs["seed"].unique())
    labels = run_labels(seeds)
    written: list[Path] = []

    for metric, ylabel, title in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for seed in seeds:
            seed_runs = runs[runs["seed"] == seed].sort_values("fraction")
            ax.plot(
                seed_runs["n_selected"].to_numpy(dtype=float),
                seed_runs[metric].to_numpy(dtype=float),
                color=SEED_COLORS.get(seed, "#333333"),
                marker=SEED_MARKERS.get(seed, "o"),
                markersize=7,
                linewidth=RUN_LINEWIDTH,
                alpha=RUN_ALPHA,
                solid_capstyle="round",
                label=labels[seed],
                zorder=2,
            )
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        std = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color="#c0392b",
            linewidth=MEAN_LINEWIDTH,
            alpha=1.0,
            label=f"Mean of {len(seeds)} runs",
            zorder=4,
        )
        has_sample_std = summary["n_seeds"].to_numpy() > 1
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            where=has_sample_std & np.isfinite(std),
            color="#c0392b",
            alpha=0.18,
            label="± 1 sample std",
            zorder=1,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x[0] * 0.85, x[-1] * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Training snapshot")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        path = fig_dir / FIGURE_NAMES[metric]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"Wrote {path}")

    # Remove the old 1x3 combined figure if it remains from earlier runs.
    legacy = fig_dir / "test_error_vs_training_snapshots.png"
    if legacy.exists():
        legacy.unlink()
    return written


def write_report(
    *,
    title: str,
    selection_note: str,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = plot_scaling(runs, summary, fig_dir)

    # Wide CSV with per-run columns for the average table.
    summary_rows = []
    seeds = sorted(int(s) for s in runs["seed"].unique())
    labels = run_labels(seeds)
    for _, row in summary.iterrows():
        record = {
            "fraction": float(row["fraction"]),
            "n_cases": int(row["n_selected"]),
            "n_fit": int(row["n_training"]),
            "n_holdout": int(row["n_validation"]),
            "n_runs": int(row["n_seeds"]),
        }
        for metric, _, _ in METRIC_SPECS:
            record[f"{metric}_mean"] = float(row[f"{metric}_mean"])
            std = row[f"{metric}_std"]
            record[f"{metric}_std"] = float(std) if pd.notna(std) else 0.0
        record["validation_rmse_mean"] = float(row["validation_rmse_mean"])
        vstd = row["validation_rmse_std"]
        record["validation_rmse_std"] = float(vstd) if pd.notna(vstd) else 0.0
        group = runs[np.isclose(runs["fraction"], row["fraction"])]
        for _, item in group.iterrows():
            seed = int(item["seed"])
            run_key = f"run{labels[seed].split()[-1]}"
            record[f"{run_key}_seed"] = seed
            record[f"{run_key}_test_rmse"] = float(item["test_rmse"])
            record[f"{run_key}_test_mae"] = float(item["test_mae"])
            record[f"{run_key}_test_nrmse_std"] = float(item["test_nrmse_std"])
            record[f"{run_key}_best_trial"] = int(item["best_trial"])
            record[f"{run_key}_validation_rmse"] = float(item["validation_rmse"])
        summary_rows.append(record)

    csv_path = out_dir / "seed_average.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_seed = runs.drop(columns=["report_path"], errors="ignore").to_dict(orient="records")
    for row in per_seed:
        row["run"] = labels[int(row["seed"])]
    (out_dir / "per_seed.json").write_text(
        json.dumps(
            {
                "updated": datetime.now().isoformat(timespec="seconds"),
                "test_set": TEST_SET,
                "selection": selection_note,
                "run_map": {labels[seed]: seed for seed in seeds},
                "rows": per_seed,
            },
            indent=2,
        )
        + "\n"
    )

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        f"# {title}",
        "",
        f"Updated {datetime.now().isoformat(timespec='seconds')}.",
        "",
        selection_note,
        "",
        f"Test scores are on {TEST_SET}. Those cases were not used for training "
        "or for selecting the checkpoint.",
        "",
        "Runs (sorted seed order): "
        + ", ".join(f"{labels[seed]} ← seed {seed}" for seed in seeds)
        + ".",
        "",
        "## Figures",
        "",
    ]
    for metric, ylabel, title_text in METRIC_SPECS:
        lines.append(f"### {title_text}")
        lines.append("")
        lines.append(f"![{ylabel}](figures/{FIGURE_NAMES[metric]})")
        lines.append("")

    lines.extend(
        [
            "## Mean test error",
            "",
            "| Snapshot | Cases | Fit / holdout | Runs | Test RMSE | Test MAE | Test NRMSE(std) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        lines.append(
            f"| {pct}% | {row['n_cases']} | {row['n_fit']} / {row['n_holdout']} | "
            f"{row['n_runs']} | "
            f"{fmt(row['test_rmse_mean'])} ± {fmt(row['test_rmse_std'])} | "
            f"{fmt(row['test_mae_mean'])} ± {fmt(row['test_mae_std'])} | "
            f"{fmt(row['test_nrmse_std_mean'])} ± {fmt(row['test_nrmse_std_std'])} |"
        )

    run_header = " | ".join(labels[seed] for seed in seeds)
    lines.extend(
        [
            "",
            "## Test RMSE by run",
            "",
            f"| Snapshot | {run_header} |",
            "|---:|" + "|".join(["---:"] * len(seeds)) + "|",
        ]
    )
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        cells = []
        for seed in seeds:
            run_key = f"run{labels[seed].split()[-1]}"
            value = row[f"{run_key}_test_rmse"]
            trial = int(row[f"{run_key}_best_trial"])
            if trial >= 0:
                cells.append(f"{fmt(value)} (trial {trial:03d})")
            else:
                cells.append(fmt(value))
        lines.append(f"| {pct}% | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Holdout / validation RMSE by run",
            "",
            f"| Snapshot | {run_header} |",
            "|---:|" + "|".join(["---:"] * len(seeds)) + "|",
        ]
    )
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        cells = [
            fmt(row[f"run{labels[seed].split()[-1]}_validation_rmse"]) for seed in seeds
        ]
        lines.append(f"| {pct}% | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            *[f"- Figure: `figures/{FIGURE_NAMES[metric]}`" for metric, _, _ in METRIC_SPECS],
            "- Mean table: `seed_average.csv`",
            "- One row per run and snapshot: `per_seed.json`",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'report.md'}")
    for path in figure_paths:
        print(f"Wrote {path}")


def write_combined_overlay(hf: pd.DataFrame, mf: pd.DataFrame, autohp: pd.DataFrame, out_dir: Path) -> None:
    """One overlay figure per metric comparing the three experiment means."""
    apply_style()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    series = (
        ("HF autoregressor", hf, "#1f4e79", "o"),
        ("MF fixed HP", mf, "#d35400", "s"),
        ("MF + autohp", autohp, "#2e8b57", "D"),
    )
    overlay_names = {
        "test_rmse": "hf_mf_test_rmse_overlay.png",
        "test_mae": "hf_mf_test_mae_overlay.png",
        "test_nrmse_std": "hf_mf_test_nrmse_std_overlay.png",
    }
    for metric, ylabel, title in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for name, runs, color, marker in series:
            summary = summarize(runs)
            x = summary["n_selected"].to_numpy(dtype=float)
            mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
            std = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                lw=MEAN_LINEWIDTH,
                ms=8,
                alpha=1.0,
                label=name,
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training snapshot size (cases)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        path = fig_dir / overlay_names[metric]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")
    legacy = fig_dir / "hf_mf_test_error_overlay.png"
    if legacy.exists():
        legacy.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which",
        nargs="+",
        choices=("hf", "mf", "autohp", "all"),
        default=["all"],
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    which = set(args.which)
    if "all" in which:
        which = {"hf", "mf", "autohp"}

    hf_runs = mf_runs = autohp_runs = None

    if "mf" in which:
        mf_runs = load_mf_runs(REPO / "train_mf_scaling_results")
        write_report(
            title="Multi-fidelity scaling (fixed hyperparameters)",
            selection_note=(
                "Each point is the trained MF model for that nested snapshot. "
                "Hyperparameters are fixed; the training-geometry holdout RMSE is "
                "reported but does not select among Optuna trials."
            ),
            runs=mf_runs,
            summary=summarize(mf_runs),
            out_dir=REPO / "train_mf_scaling_report",
        )

    if "autohp" in which:
        autohp_runs = load_mf_runs(REPO / "train_mf_scaling_autohp_results")
        write_report(
            title="Multi-fidelity scaling (batch size / LR Optuna)",
            selection_note=(
                "Each point is the Optuna winner for that nested snapshot. "
                "Optuna minimized training-geometry holdout RMSE; batch size and "
                "learning rate were searched around the scaling-law centers."
            ),
            runs=autohp_runs,
            summary=summarize(autohp_runs),
            out_dir=REPO / "train_mf_scaling_autohp_report",
        )

    if "hf" in which:
        hf_runs = load_hf_runs(REPO / "train_hf_scaling_results")
        write_report(
            title="High-fidelity autoregressor scaling",
            selection_note=(
                "Each point is the Optuna winner for that nested snapshot. "
                "Optuna minimized training-geometry holdout RMSE inside the snapshot."
            ),
            runs=hf_runs,
            summary=summarize(hf_runs),
            out_dir=REPO / "train_hf_scaling_report",
        )

    # Combined overlay when all three are available (load any missing).
    if hf_runs is None and (REPO / "train_hf_scaling_results").exists():
        try:
            hf_runs = load_hf_runs(REPO / "train_hf_scaling_results")
        except FileNotFoundError:
            hf_runs = None
    if mf_runs is None and (REPO / "train_mf_scaling_results").exists():
        try:
            mf_runs = load_mf_runs(REPO / "train_mf_scaling_results")
        except FileNotFoundError:
            mf_runs = None
    if autohp_runs is None and (REPO / "train_mf_scaling_autohp_results").exists():
        try:
            autohp_runs = load_mf_runs(REPO / "train_mf_scaling_autohp_results")
        except FileNotFoundError:
            autohp_runs = None
    if hf_runs is not None and mf_runs is not None and autohp_runs is not None:
        write_combined_overlay(
            hf_runs,
            mf_runs,
            autohp_runs,
            REPO / "train_hf_scaling_report",
        )
        # also drop a copy next to MF reports for convenience
        write_combined_overlay(
            hf_runs,
            mf_runs,
            autohp_runs,
            REPO / "train_mf_scaling_report",
        )

    # Refresh notebook prose so they mention the correct test set.
    for nb_name, label in (
        ("mf_scaling_analysis.ipynb", "lhs-free-stream generalized geometries"),
        ("mf_scaling_analysis_with_hpo.ipynb", "lhs-free-stream generalized geometries"),
    ):
        path = REPO / nb_name
        if not path.is_file():
            continue
        text = path.read_text()
        text = text.replace(
            "The final test geometries (`dataset/new_geom`)",
            f"The final test geometries ({label} under `dataset/new_geom/lhs-free-stream`)",
        )
        path.write_text(text)

    print("Done.")


if __name__ == "__main__":
    main()
