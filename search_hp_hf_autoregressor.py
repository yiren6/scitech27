"""Optuna search for the HF→HF VortexNet autoregressor.

Each trial trains GNN4 on the same 80/20 split of dataset/train_set used by
train_hf_autoregressor.py. Optuna minimizes holdout RMSE on that
training-geometry split. The unseen dataset/new_geom wings are scored and
saved for every trial, and they are not used to choose hyperparameters.

Every trial writes:

    optuna_results/trial_XXX/model.pt
    optuna_results/trial_XXX/report.json

report.json contains the training report (fit split, holdout, learning curve)
and the test-geometry report. The study summary and the best trial are copied
to optuna_results/study_summary.json and optuna_results/best/.

The default is a test run (one trial, one epoch, a few cases). The full
search uses every training case:

    python VortexNet/hf_autoregressor/search_hp_hf_autoregressor.py
    python VortexNet/hf_autoregressor/search_hp_hf_autoregressor.py --mode full
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import optuna
import torch

REPO_ROOT = Path(__file__).resolve().parents[0]
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_hf_autoregressor import (  # noqa: E402
    _Tee,
    _format_metrics,
    build_report,
    evaluate_prepared,
    fit_hf_model,
    make_model,
    prepare_hf_data,
    write_train_set_split,
)


# Values left unset on the command line are filled from the selected mode.
_MODE_PRESETS = {
    "test": {
        "n_trials": 1,
        "epochs": 1,
        "patience": 1,
        "scheduler_patience": 1,
        "batch_size": 4,
        "max_cases": 16,
        "output_dir": str(PACKAGE_DIR / "optuna_results_test"),
        "study_name": "hf_autoregressor_test",
    },
    "full": {
        "n_trials": 100,
        "epochs": 3000,
        "patience": 30,
        "scheduler_patience": 4,
        "batch_size": 64,
        "max_cases": None,
        "output_dir": str(PACKAGE_DIR / "optuna_results"),
        "study_name": "hf_autoregressor",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("test", "full"),
        default="test",
        help=(
            "test: one trial and one epoch on a few cases (default). "
            "full: full-batch Optuna search on every training case."
        ),
    )
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "dataset" / "train_set"))
    parser.add_argument("--test-dir", default=str(REPO_ROOT / "dataset" / "new_geom"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to optuna_results_test or optuna_results for --mode full.",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--clip-value", type=float, default=30.0)
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    parser.add_argument("--ref-re", type=float, default=1e7)
    parser.add_argument("--ref-length", type=float, default=0.435762)
    parser.add_argument("--root-chord-ref", type=float, default=0.6536436)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap training-geometry cases. The test mode preset is 16; full uses all cases.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def apply_run_mode(args: argparse.Namespace) -> None:
    """Fill any omitted search settings from the test or full preset."""
    preset = _MODE_PRESETS[args.mode]
    # max_cases is None both when omitted and when the full preset uses every case.
    # A user-supplied positive value always wins. The test preset applies only when
    # the flag was left unset, which we detect by mode plus the sentinel below.
    explicit_max_cases = args.max_cases
    for key, value in preset.items():
        if key == "max_cases":
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.mode == "test" and explicit_max_cases is None:
        args.max_cases = preset["max_cases"]
    if getattr(args, "study_name", None) is None:
        args.study_name = preset["study_name"]


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """GNN4 and Adam settings. Physics-loss weights are omitted: this model is HF only."""
    return {
        "hidden_channels": trial.suggest_int("hidden_channels", 8, 48),
        "heads": trial.suggest_int("heads", 1, 5),
        "hop": trial.suggest_int("hop", 3, 15),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    }


def spec_from_trial(args, hyperparams: dict) -> Namespace:
    return Namespace(
        hidden_channels=hyperparams["hidden_channels"],
        heads=hyperparams["heads"],
        hop=hyperparams["hop"],
        alpha=hyperparams["alpha"],
        dropout=hyperparams["dropout"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        noise=args.noise,
        clip_value=args.clip_value,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        penalty_weight=args.penalty_weight,
        seed=args.seed,
        test_size=args.test_size,
    )


def save_checkpoint(path: Path, trained: dict) -> None:
    torch.save(
        {
            "model_state": trained["best_state"],
            "model_kwargs": trained["model_kwargs"],
            "best_train_smoothl1": trained["best_train_smoothl1"],
        },
        path,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def trial_record(trial: optuna.trial.FrozenTrial) -> dict:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": dict(trial.user_attrs),
    }


def write_study_summary(study: optuna.Study, out_dir: Path) -> None:
    completed = [trial for trial in study.trials if trial.value is not None]
    best = None
    if completed:
        winner = min(completed, key=lambda trial: trial.value)
        best = trial_record(winner)
    summary = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "study": study.study_name,
        "direction": "minimize",
        "objective": "training_geometry_holdout.rmse",
        "objective_note": (
            "Hyperparameters are chosen by RMSE on the random 20% holdout of "
            "dataset/train_set. dataset/new_geom is reported for every trial "
            "and is not the Optuna objective."
        ),
        "mode": study.user_attrs.get("mode"),
        "n_trials": len(study.trials),
        "best_trial": best,
        "trials": [trial_record(trial) for trial in study.trials],
    }
    write_json(out_dir / "study_summary.json", summary)
    if best is not None:
        source = out_dir / f"trial_{best['number']:03d}"
        destination = out_dir / "best"
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)


def score_and_save(args, prepared, device, trial_dir: Path, trained, spec, status: str) -> dict:
    model = make_model(spec, device)
    model.load_state_dict(trained["best_state"])
    fit_metrics, holdout_metrics, test_metrics = evaluate_prepared(
        model, prepared, device, args.batch_size
    )
    report = build_report(spec, prepared, trained, fit_metrics, holdout_metrics, test_metrics)
    report["status"] = status
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(trial_dir / "model.pt", trained)
    write_json(trial_dir / "report.json", report)
    print(
        f"Saved {trial_dir.name} ({status})  "
        f"holdout {_format_metrics(holdout_metrics)}"
    )
    print(f"  test geometry {_format_metrics(test_metrics)}")
    return report


def search(args) -> None:
    apply_run_mode(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "search_log.txt"
    log_file = log_path.open("w")
    sys.stdout = _Tee(log_file, sys.__stdout__)
    sys.stderr = _Tee(log_file, sys.__stderr__)

    device = torch.device(args.device)
    print(f"HF autoregressor Optuna search  {datetime.now().isoformat(timespec='seconds')}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(
        f"Trials: {args.n_trials}   epochs/trial: {args.epochs}   "
        f"batch size: {args.batch_size}   max cases: {args.max_cases}"
    )

    prepared = prepare_hf_data(args)
    split_json, split_text = write_train_set_split(out_dir, prepared["train_set_split"])
    print(f"Wrote {split_json}")
    print(f"Wrote {split_text}")

    def objective(trial: optuna.Trial) -> float:
        hyperparams = suggest_hyperparameters(trial)
        spec = spec_from_trial(args, hyperparams)
        trial_dir = out_dir / f"trial_{trial.number:03d}"
        print("\n" + "-" * 72)
        print(f"Trial {trial.number}  {hyperparams}")
        partial: dict = {}
        try:
            try:
                trained = fit_hf_model(spec, prepared, device, trial=trial, partial=partial)
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
            report = score_and_save(
                args, prepared, device, trial_dir, trained, spec, status
            )
            trial.set_user_attr("status", status)
            trial.set_user_attr("holdout_rmse", report["training_geometry_holdout"]["rmse"])
            trial.set_user_attr("holdout_nrmse_std", report["training_geometry_holdout"]["nrmse_std"])
            trial.set_user_attr("test_rmse", report["test_report"]["rmse"])
            trial.set_user_attr("test_nrmse_std", report["test_report"]["nrmse_std"])
            trial.set_user_attr("report", str(trial_dir / "report.json"))
            trial.set_user_attr("model", str(trial_dir / "model.pt"))
            del trained
            gc.collect()
            if status == "pruned":
                raise optuna.TrialPruned()
            return report["training_geometry_holdout"]["rmse"]
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            write_json(
                trial_dir / "report.json",
                {
                    "status": "failed",
                    "error": repr(exc),
                    "hyperparameters": hyperparams,
                },
            )
            raise

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    study.set_user_attr("mode", args.mode)

    def persist(_study, _trial):
        write_study_summary(_study, out_dir)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(Exception,),
        callbacks=[persist],
    )
    write_study_summary(study, out_dir)

    print("\n" + "=" * 72)
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        print("No trial finished with a holdout score.")
    else:
        best = min(completed, key=lambda trial: trial.value)
        print(f"Best trial {best.number}  holdout RMSE {best.value:.6f}")
        print(f"Hyperparameters: {best.params}")
        print(
            "Test-geometry RMSE "
            f"{best.user_attrs.get('test_rmse')}   "
            f"NRMSE(std) {best.user_attrs.get('test_nrmse_std')}"
        )
        print(f"Best checkpoint: {out_dir / 'best' / 'model.pt'}")
    print(f"Study summary: {out_dir / 'study_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        search(parse_args())
    except Exception:
        traceback.print_exc()
        raise
