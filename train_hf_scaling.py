"""Train and tune the HF autoregressor at several training-set sizes.

The 6%, 16%, 20%, 40%, 60%, 80%, and 100% subsets are nested random samples
of dataset/train_set. Each subset gets its own Optuna search using a random
training-geometry holdout for selection. dataset/new_geom is only used for
final reporting.

Run a short smoke test with:

    python train_hf_scaling.py --n-trials 1 --epochs 1 --max-cases 16
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from search_hp_hf_autoregressor import (  # noqa: E402
    save_checkpoint,
    spec_from_trial,
    suggest_hyperparameters,
    trial_record,
    write_json,
)
from train_hf_autoregressor import (  # noqa: E402
    _Tee,
    _format_metrics,
    build_hf_graphs,
    build_report,
    evaluate_prepared,
    fit_hf_model,
    load_cases,
    make_model,
)


FRACTIONS = (0.06, 0.16, 0.2, 0.4, 0.6, 0.8, 1.0)
SPLIT_COLUMNS = (
    "fraction",
    "split",
    "dataset_index",
    "graph_index",
    "source_file",
    "case_index",
    "test_number",
    "mach",
    "alpha",
    "Re",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "dataset" / "train_set"))
    parser.add_argument("--test-dir", default=str(REPO_ROOT / "dataset" / "new_geom"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "train_hf_scaling_results"),
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        choices=FRACTIONS,
        default=list(FRACTIONS),
        help="Training-set fractions to run (default: all fractions).",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--clip-value", type=float, default=30.0)
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    parser.add_argument("--ref-re", type=float, default=1e7)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def _case_record(case, split: str, fraction: float, graph_index: int) -> dict:
    return {
        "fraction": fraction,
        "split": split,
        "dataset_index": int(case.dataset_index),
        "graph_index": graph_index,
        "source_file": case.source_file,
        "case_index": int(case.case_index),
        "test_number": case.test_number,
        "mach": case.mach,
        "alpha": case.alpha,
        "Re": case.Re,
    }


def _prepare_fraction(
    fraction: float,
    cases: list,
    graphs: list,
    sources: list[str],
    test_graphs: list,
    test_sources: list[str],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    count = max(1, round(len(cases) * fraction))
    if count < 2:
        raise ValueError(
            f"The {fraction:.0%} subset contains only {count} case(s); "
            "at least two are required for a fit/holdout split."
        )
    subset_indices = np.random.default_rng(args.seed).permutation(len(cases))[:count]
    split_seed = args.seed + round(fraction * 100)
    fit_indices, holdout_indices = train_test_split(
        subset_indices.tolist(),
        test_size=args.test_size,
        random_state=split_seed,
    )

    fit_cases = [cases[index] for index in fit_indices]
    holdout_cases = [cases[index] for index in holdout_indices]
    fit_graphs = [graphs[index] for index in fit_indices]
    holdout_graphs = [graphs[index] for index in holdout_indices]
    fit_sources = [sources[index] for index in fit_indices]
    holdout_sources = [sources[index] for index in holdout_indices]

    records = [
        _case_record(case, "training", fraction, graph_index)
        for graph_index, case in enumerate(fit_cases)
    ] + [
        _case_record(case, "test", fraction, graph_index)
        for graph_index, case in enumerate(holdout_cases)
    ]
    records.sort(key=lambda row: (row["dataset_index"], row["split"]))

    prepared = {
        "scaler": None,
        "fit_graphs": fit_graphs,
        "fit_sources": fit_sources,
        "holdout_graphs": holdout_graphs,
        "holdout_sources": holdout_sources,
        "test_graphs": test_graphs,
        "test_sources": test_sources,
        "train_set_split": {
            "seed": split_seed,
            "test_size": args.test_size,
            "n_training": len(fit_cases),
            "n_test": len(holdout_cases),
        },
    }
    return prepared, records


def _write_split_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _write_study_summary(study: optuna.Study, out_dir: Path) -> dict | None:
    completed = [trial for trial in study.trials if trial.value is not None]
    best = min(completed, key=lambda trial: trial.value) if completed else None
    summary = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "study": study.study_name,
        "direction": "minimize",
        "objective": "training_geometry_holdout.rmse",
        "n_trials": len(study.trials),
        "best_trial": trial_record(best) if best is not None else None,
        "trials": [trial_record(trial) for trial in study.trials],
    }
    write_json(out_dir / "study_summary.json", summary)
    return best


def _run_fraction(
    fraction: float,
    prepared: dict,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict | None:
    percent = round(fraction * 100)
    fraction_dir = out_dir / f"fraction_{percent:02d}"
    fraction_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"hf_scaling_{percent:02d}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed + percent),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        storage=f"sqlite:///{fraction_dir / 'study.db'}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        hyperparams = suggest_hyperparameters(trial)
        spec = spec_from_trial(args, hyperparams)
        trial_dir = fraction_dir / f"trial_{trial.number:03d}"
        partial: dict = {}
        try:
            trained = fit_hf_model(
                spec, prepared, device, trial=trial, partial=partial
            )
            status = "complete"
        except optuna.TrialPruned:
            trained = partial
            status = "pruned"
            if "best_state" not in trained:
                raise

        trained.pop("model", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = make_model(spec, device)
        model.load_state_dict(trained["best_state"])
        fit_metrics, holdout_metrics, test_metrics = evaluate_prepared(
            model, prepared, device, args.batch_size
        )
        report = build_report(
            spec, prepared, trained, fit_metrics, holdout_metrics, test_metrics
        )
        report["status"] = status
        report["training_fraction"] = fraction
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(trial_dir / "model.pt", trained)
        write_json(trial_dir / "report.json", report)
        trial.set_user_attr("status", status)
        trial.set_user_attr("holdout_rmse", holdout_metrics["rmse"])
        trial.set_user_attr("test_rmse", test_metrics["rmse"])
        trial.set_user_attr("report", str(trial_dir / "report.json"))
        trial.set_user_attr("model", str(trial_dir / "model.pt"))
        print(
            f"Fraction {percent}% trial {trial.number} ({status}) "
            f"holdout {_format_metrics(holdout_metrics)}"
        )
        print(f"  test geometry {_format_metrics(test_metrics)}")
        if status == "pruned":
            raise optuna.TrialPruned()
        return holdout_metrics["rmse"]

    remaining_trials = max(0, args.n_trials - len(study.trials))
    study.optimize(objective, n_trials=remaining_trials)
    best = _write_study_summary(study, fraction_dir)
    if best is None:
        print(f"No completed trial for the {percent}% subset.")
        return None

    best_trial_dir = fraction_dir / f"trial_{best.number:03d}"
    best_dir = fraction_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_trial_dir / "model.pt", best_dir / "model.pt")
    shutil.copy2(best_trial_dir / "report.json", best_dir / "report.json")
    with (best_trial_dir / "report.json").open() as handle:
        report = json.load(handle)
    return {
        "fraction": fraction,
        "n_cases": len(prepared["fit_graphs"]) + len(prepared["holdout_graphs"]),
        "n_fit": len(prepared["fit_graphs"]),
        "n_holdout": len(prepared["holdout_graphs"]),
        "best_trial": best.number,
        "holdout_rmse": report["training_geometry_holdout"]["rmse"],
        "holdout_mae": report["training_geometry_holdout"]["mae"],
        "test_rmse": report["test_report"]["rmse"],
        "test_mae": report["test_report"]["mae"],
        "test_nrmse_std": report["test_report"]["nrmse_std"],
        "hyperparameters": json.dumps(best.params, sort_keys=True),
        "report": str(best_dir / "report.json"),
        "checkpoint": str(best_dir / "model.pt"),
    }


def run(args: argparse.Namespace) -> None:
    if args.n_trials < 1 or args.epochs < 1:
        raise ValueError("--n-trials and --epochs must be positive.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not args.fractions or len(set(args.fractions)) != len(args.fractions):
        raise ValueError("--fractions must contain one or more unique fractions.")
    fractions = tuple(args.fractions)
    selected_percentages = {round(fraction * 100) for fraction in fractions}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "train_log.txt").open("a")
    sys.stdout = _Tee(log_file, sys.__stdout__)
    sys.stderr = _Tee(log_file, sys.__stderr__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"HF autoregressor scaling run {datetime.now().isoformat(timespec='seconds')}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    cases = load_cases(Path(args.train_dir), validation=False, max_cases=args.max_cases)
    if len(cases) < 2:
        raise ValueError("At least two training cases are required.")
    for dataset_index, case in enumerate(cases):
        case.dataset_index = dataset_index
    print("\nBuilding all training graphs")
    graphs, sources = build_hf_graphs(cases, args.ref_re)

    print("\nLoading test geometries")
    test_cases = load_cases(Path(args.test_dir), validation=True)
    print("Building test-geometry graphs")
    test_graphs, test_sources = build_hf_graphs(test_cases, args.ref_re)

    all_records: list[dict] = []
    prepared_by_fraction = {}
    for fraction in fractions:
        prepared, records = _prepare_fraction(
            fraction, cases, graphs, sources, test_graphs, test_sources, args
        )
        prepared_by_fraction[fraction] = prepared
        all_records.extend(records)
    split_path = out_dir / "train_set_split.csv"
    if split_path.exists():
        with split_path.open(newline="") as handle:
            previous_records = list(csv.DictReader(handle))
        all_records.extend(
            row
            for row in previous_records
            if round(float(row["fraction"]) * 100) not in selected_percentages
        )
    all_records.sort(
        key=lambda row: (
            float(row["fraction"]),
            int(row["dataset_index"]),
            row["split"],
        )
    )
    _write_split_csv(split_path, all_records)
    print(f"Wrote snapshot assignment CSV: {split_path}")

    results = []
    for fraction in fractions:
        print("\n" + "=" * 72)
        print(f"Searching hyperparameters with {round(fraction * 100)}% of training cases")
        result = _run_fraction(
            fraction, prepared_by_fraction[fraction], args, device, out_dir
        )
        if result is not None:
            results.append(result)

    results_path = out_dir / "scaling_results.csv"
    summary_path = out_dir / "scaling_results.json"
    if summary_path.exists():
        with summary_path.open() as handle:
            previous_summary = json.load(handle)
        results.extend(
            result
            for result in previous_summary.get("results", [])
            if round(float(result["fraction"]) * 100) not in selected_percentages
        )
    results.sort(key=lambda result: result["fraction"])

    if results:
        with results_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    write_json(
        summary_path,
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "subset_sampling": "Nested random subsets; seed-controlled.",
            "objective": "training_geometry_holdout.rmse",
            "test_set_used_for_selection": False,
            "results": results,
        },
    )
    print(f"\nWrote {results_path}")
    print(f"Wrote {out_dir / 'scaling_results.json'}")
    for result in results:
        print(
            f"{result['fraction']:.0%} data: "
            f"holdout RMSE {result['holdout_rmse']:.6f}; "
            f"test RMSE {result['test_rmse']:.6f}; "
            f"test MAE {result['test_mae']:.6f}"
        )


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        traceback.print_exc()
        raise
