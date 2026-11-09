"""Hyperparameter reports for the HF scaling studies, plus a 3-seed average.

Each ``train_hf_scaling_results_seed*/fraction_*`` folder is one Optuna study.
This writes the same diagnostic report used for the October 26 searches
(``hp_search_report/``) and a cross-seed summary at ``train_hf_scaling_report/``.

Every figure uses a font size of at least 18 pt.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

REPO = Path(__file__).resolve().parent
VORTEX_ROOT = Path("/Users/yirenshen/Desktop/MF-VortexNet")
sys.path.insert(0, str(VORTEX_ROOT))
sys.path.insert(0, str(VORTEX_ROOT / "VortexNet"))

from VortexNet.hf_autoregressor import report_optuna_search as R  # noqa: E402

FONT = 18
SEEDS = (7, 27, 3407)
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
SUMMARY_DIR = REPO / "train_hf_scaling_report"

_orig_subplots = plt.subplots
_orig_text = Axes.text


def _subplots(*args, **kwargs):
    if "figsize" in kwargs:
        width, height = kwargs["figsize"]
        kwargs["figsize"] = (width * 1.7, height * 1.55)
    return _orig_subplots(*args, **kwargs)


def _text(self, x, y, s, *args, **kwargs):
    size = kwargs.get("fontsize")
    if size is not None and size < FONT:
        kwargs["fontsize"] = FONT
    return _orig_text(self, x, y, s, *args, **kwargs)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": FONT,
            "axes.titlesize": FONT + 2,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "figure.titlesize": FONT + 2,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def plot_top5_comparison(study: dict, fig_dir: Path) -> Path:
    """Bar chart without in-bar numerals. At 18 pt those labels collide."""
    top = study["top"]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    labels = [f"Trial {trial['number']:03d}" for trial in top]
    x = np.arange(len(top))
    width = 0.24
    series = (
        ("Fit", [trial["fit_rmse"] for trial in top], "#7f8c8d"),
        ("Holdout (selection)", [trial["holdout_rmse"] for trial in top], R.COMPLETE_COLOR),
        ("Test geometry", [trial["test_rmse"] for trial in top], R.TEST_COLOR),
    )
    for offset, (name, values, color) in enumerate(series):
        ax.bar(x + (offset - 1) * width, values, width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (ΔCp)")
    ax.set_title(f"Top {len(top)} completed trials by holdout RMSE")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(y=0.18)
    return R._save(fig, fig_dir / "04_top5_metric_comparison.png")


def plot_field_grid(fields, model_preds, err_lim, fig_path, figure_title):
    figure_title = figure_title.replace(
        "training-geometry holdout",
        "reference training geometries",
    )
    return _orig_field_grid(fields, model_preds, err_lim, fig_path, figure_title)


_orig_field_grid = R.plot_field_grid


def install_style() -> None:
    plt.subplots = _subplots
    Axes.text = _text
    R._style = apply_style
    R._set_field_font = apply_style
    R.plot_top5_comparison = plot_top5_comparison
    R.plot_field_grid = plot_field_grid
    apply_style()


def fraction_dir(seed: int, fraction: float) -> Path:
    pct = int(round(fraction * 100))
    return REPO / f"train_hf_scaling_results_seed{seed}" / f"fraction_{pct}"


def polish_report(path: Path, seed: int, fraction: float, study: dict) -> None:
    best = study["best"] or {}
    split = best.get("split") or {}
    n_fit = int(split.get("n_fit"))
    n_holdout = int(split.get("n_holdout"))
    n_cases = n_fit + n_holdout
    pct = int(round(fraction * 100))
    fit_pct = int(round(100 * n_fit / n_cases))
    hold_pct = 100 - fit_pct
    text = path.read_text()
    text = text.replace(
        f"from `{study['results_dir'].name}`.",
        (
            f"from `{study['results_dir']}`. "
            f"This study is the seed-{seed} nested snapshot that keeps "
            f"{pct}% of `dataset/train_set`."
        ),
    )
    text = text.replace(
        "- Mode recorded in the study: `None`",
        (
            f"- Random seed for the nested snapshot and the trial training: **{seed}**\n"
            f"- Training snapshot: **{pct}%** of the 600 training cases "
            f"({n_cases} cases; {n_fit} fit / {n_holdout} holdout, "
            f"about {fit_pct}/{hold_pct})"
        ),
    )
    text = text.replace(
        "Fit (80% training geometries)",
        f"Fit ({fit_pct}% of this snapshot)",
    )
    old_fields = (
        "The same ten cases are plotted in both the full-set and the 55° NACA 0010 "
        "reports. The five training-geometry cases are the intersection of the two "
        "studies' holdout splits. The five test-geometry cases come from "
        "`dataset/new_geom` and were not used to train or to select either model. "
        "Each row is one case. Columns are predicted ΔCp and error "
        "(ground truth − prediction) for the top two completed trials. "
        "Predicted fields use that case's ground-truth ΔCp range. Error uses one "
        "symmetric scale per figure, shared across both reports."
    )
    new_fields = (
        "The ten cases are the same reference cases used in the October 26 full-set "
        "and 55° NACA 0010 reports, so snapshots can be compared. The five "
        "training-geometry cases are sweep 55° NACA 0010; a given case is not "
        "necessarily inside this snapshot's holdout. The five test-geometry cases "
        "come from `dataset/new_geom` and were not used to train or to select the "
        "model. Each row is one case. Columns are predicted ΔCp and error "
        "(ground truth − prediction) for the top two completed trials. Predicted "
        "fields use that case's ground-truth ΔCp range. Error uses one symmetric "
        "scale for this figure."
    )
    text = text.replace(old_fields, new_fields)
    text = text.replace(
        "### Training-geometry holdout",
        "### Reference training-geometry cases",
    )
    path.write_text(text)


def write_hpo_reports(field_sets: dict, device) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for fraction in FRACTIONS:
            results_dir = fraction_dir(seed, fraction)
            pct = int(round(fraction * 100))
            title = f"Seed {seed}, {pct}% training snapshot"
            print(f"\n=== {title} ===", flush=True)
            study = R.load_study(results_dir, title, f"seed{seed}_fraction_{pct}")
            study["output_root"] = None
            R.attach_field_plots([study], field_sets, device)
            report_path = R.report_study(study, None)
            polish_report(report_path, seed, fraction, study)
            best = study["best"]
            split = best["split"]
            rows.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "n_cases": int(split["n_fit"]) + int(split["n_holdout"]),
                    "n_fit": int(split["n_fit"]),
                    "n_holdout": int(split["n_holdout"]),
                    "best_trial": int(best["number"]),
                    "holdout_rmse": float(best["holdout_rmse"]),
                    "holdout_mae": float(best["holdout_mae"]),
                    "test_rmse": float(best["test_rmse"]),
                    "test_mae": float(best["test_mae"]),
                    "test_nrmse_std": float(best["test_nrmse_std"]),
                    "n_complete": len(study["completed"]),
                    "n_trials": len(study["trials"]),
                    "report": str(report_path),
                }
            )
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def write_average_report(rows: list[dict]) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = SUMMARY_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_fraction: dict[float, list[dict]] = {}
    for row in rows:
        by_fraction.setdefault(row["fraction"], []).append(row)
    fractions = sorted(by_fraction)
    summary_rows = []
    for fraction in fractions:
        group = by_fraction[fraction]
        record = {
            "fraction": fraction,
            "n_cases": group[0]["n_cases"],
            "n_fit": group[0]["n_fit"],
            "n_holdout": group[0]["n_holdout"],
        }
        for key in ("test_rmse", "test_mae", "test_nrmse_std", "holdout_rmse"):
            mean, std = _mean_std([item[key] for item in group])
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = std
        for item in group:
            record[f"seed{item['seed']}_test_rmse"] = item["test_rmse"]
            record[f"seed{item['seed']}_test_mae"] = item["test_mae"]
            record[f"seed{item['seed']}_test_nrmse_std"] = item["test_nrmse_std"]
            record[f"seed{item['seed']}_best_trial"] = item["best_trial"]
        summary_rows.append(record)

    csv_path = SUMMARY_DIR / "seed_average.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (SUMMARY_DIR / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    apply_style()
    # This figure is drawn directly, so bypass the global figsize scaler.
    plt.subplots = _orig_subplots
    x = np.array([row["n_cases"] for row in summary_rows], dtype=float)
    tick_labels = [
        f"{int(round(row['fraction'] * 100))}%\n{row['n_cases']} cases"
        for row in summary_rows
    ]
    panels = (
        ("test_rmse", "Test RMSE (ΔCp)", "Unseen-geometry RMSE"),
        ("test_mae", "Test MAE (ΔCp)", "Unseen-geometry MAE"),
        ("test_nrmse_std", "Test NRMSE (std)", "Unseen-geometry NRMSE"),
    )
    seed_style = {
        7: {"color": "#1f4e79", "marker": "o"},
        27: {"color": "#d35400", "marker": "s"},
        3407: {"color": "#2e8b57", "marker": "D"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 7.4))
    for ax, (key, ylabel, title) in zip(axes, panels):
        mean = np.array([row[f"{key}_mean"] for row in summary_rows])
        std = np.array([row[f"{key}_std"] for row in summary_rows])
        for seed, style in seed_style.items():
            series = [row[f"seed{seed}_{key}"] for row in summary_rows]
            ax.plot(
                x,
                series,
                color=style["color"],
                marker=style["marker"],
                ms=9,
                lw=1.6,
                alpha=0.9,
                label=f"Seed {seed}",
            )
        ax.plot(x, mean, color="#c0392b", lw=2.6, label="Mean of 3 seeds", zorder=4)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="#c0392b",
            alpha=0.18,
            label="± 1 sample std",
            zorder=1,
        )
        ax.set_xticks(x, tick_labels)
        ax.set_xlabel("Training snapshot")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(x[0] - 30, x[-1] + 30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    figure_path = fig_dir / "test_error_vs_training_snapshots.png"
    fig.savefig(figure_path)
    plt.close(fig)
    plt.subplots = _subplots

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# Training-snapshot scaling, averaged over three seeds",
        "",
        "Each point is the Optuna winner for that snapshot. Optuna minimized "
        "holdout RMSE inside the snapshot. The score plotted here is that winner's "
        "error on the 44 unseen wings in `dataset/new_geom`. Those wings were not "
        "used for training or for choosing the trial.",
        "",
        "The three runs use nested training snapshots with seeds **7**, **27**, and "
        "**3407**. A snapshot keeps 20%, 40%, 60%, or 80% of the 600 training cases. "
        "Inside each snapshot, 30% of the kept cases are a holdout (`test_size` 0.3) "
        "and the rest are the fit split. The curve is the mean across the three seeds. "
        "The band is the sample standard deviation (n = 3).",
        "",
        "Per-snapshot hyperparameter reports, with the search history, top trials, "
        "and pressure-field snapshots, are in "
        "`train_hf_scaling_results_seed{7,27,3407}/fraction_{20,40,60,80}/hp_search_report/`.",
        "",
        "![Test error versus training snapshot](figures/test_error_vs_training_snapshots.png)",
        "",
        "## Mean test error",
        "",
        "| Snapshot | Cases | Fit / holdout | Test RMSE | Test MAE | Test NRMSE(std) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        lines.append(
            f"| {pct}% | {row['n_cases']} | {row['n_fit']} / {row['n_holdout']} | "
            f"{fmt(row['test_rmse_mean'])} ± {fmt(row['test_rmse_std'])} | "
            f"{fmt(row['test_mae_mean'])} ± {fmt(row['test_mae_std'])} | "
            f"{fmt(row['test_nrmse_std_mean'])} ± {fmt(row['test_nrmse_std_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Test RMSE by seed",
            "",
            "The trial number is the holdout-RMSE winner inside that study.",
            "",
            "| Snapshot | Seed 7 | Seed 27 | Seed 3407 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        lines.append(
            f"| {pct}% | {fmt(row['seed7_test_rmse'])} (trial {row['seed7_best_trial']:03d}) | "
            f"{fmt(row['seed27_test_rmse'])} (trial {row['seed27_best_trial']:03d}) | "
            f"{fmt(row['seed3407_test_rmse'])} (trial {row['seed3407_best_trial']:03d}) |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- Figure: `figures/test_error_vs_training_snapshots.png`",
            "- Mean table: `seed_average.csv`",
            "- One row per seed and snapshot: `per_seed.json`",
            "",
        ]
    )
    report_path = SUMMARY_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")
    return report_path


def main() -> None:
    install_style()
    import torch

    device = torch.device("cpu")
    print("Loading shared pressure-field cases", flush=True)
    field_sets = R.load_shared_field_cases()
    rows = write_hpo_reports(field_sets, device)
    write_average_report(rows)


if __name__ == "__main__":
    main()
