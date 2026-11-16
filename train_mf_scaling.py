"""Run nested multi-fidelity scaling experiments.

For each seed and fraction of ``dataset/train_set``:

1. Draw a nested random subset of training-geometry snapshots.
2. Split that subset 70/30 into training and validation cases.
3. Run Optuna (~30 trials) over batch size and learning rate centered on
   the scaling-law values
   ``B★ = B_ref·N/N_ref`` and ``α★ = α_ref·(N/N_ref)^(-0.5)``,
   where ``N`` is the fit-set size. Selection uses validation RMSE only.
4. Score the best trial on all snapshots in ``dataset/new_geom``.

The model, parser, and training procedure are imported from cnfvortexnetff;
this runner does not modify that project.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import json
import os
import pickle
import random
import shutil
import sys
import types
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parent
CNF_ROOT = Path("/home/yiren/cnfvortexnetff")
sys.path.insert(0, str(CNF_ROOT / "VortexNet"))
sys.path.insert(0, str(CNF_ROOT))

from VortexNet import DataParser, GNN4, train_model_k_fold  # noqa: E402


SEEDS = (7, 27, 3407, 207, 353)
FRACTIONS = (0.06, 0.10, 0.16, 0.20, 0.40, 0.60, 0.80, 1.0)

# Fixed model / physics settings. --learning-rate is α_ref for scaling.
DEFAULT_HIDDEN_CHANNELS = 63
DEFAULT_HEADS = 4
DEFAULT_PENALTY_WEIGHT = 0.025689113034518467
DEFAULT_DROPOUT = 0.08504544001617964
DEFAULT_LAMBDA = 5.8267381175451374e-05
DEFAULT_ALPHA = 0.009705552281881635
DEFAULT_HOP = 10
DEFAULT_MAX_PHY_LOSS = 0.4175220124520909
DEFAULT_NOISE = 0.03
DEFAULT_CLIP_VALUE = 30.0
DEFAULT_LEARNING_RATE = 1e-3  # α_ref
DEFAULT_DECAY = 5e-6
DEFAULT_N_REF = 420
DEFAULT_BATCH_SIZE_REF = 32
DEFAULT_N_TRIALS = 30

REF_RE = 10**7
REF_LENGTH = 0.435762
ROOT_CHORD_REF = 0.6536436
CV_RANDOM_STATE = 3407
SPLIT_COLUMNS = (
    "seed",
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
CV_SPLIT_COLUMNS = (
    "seed",
    "fraction",
    "fold",
    "role",
    "dataset_index",
    "source_file",
    "case_index",
    "test_number",
    "mach",
    "alpha",
    "Re",
)

TRAIN_DIR = REPO_ROOT / "dataset" / "train_set"
TEST_DIR = REPO_ROOT / "dataset" / "new_geom"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        default=str(TRAIN_DIR),
        help="Directory containing training-geometry MF pickle datasets.",
    )
    parser.add_argument(
        "--test-dir",
        default=str(TEST_DIR),
        help="Directory containing the held-out test-geometry validation pickles.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "train_mf_scaling_autohp_results"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        choices=FRACTIONS,
        default=list(FRACTIONS),
        help="Fractions of dataset/train_set to sample (nested per seed).",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="K-fold CV folds used by the CNF trainer on the 70%% fit set.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of each nested subset held out for Optuna selection.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Optuna trials per seed/fraction searching batch size and learning rate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    scale = parser.add_argument_group("scaling-law reference centers")
    scale.add_argument(
        "--n-ref",
        type=int,
        default=DEFAULT_N_REF,
        help="Reference fit-set size N_ref for B★ and α★.",
    )
    scale.add_argument(
        "--batch-size-ref",
        type=int,
        default=DEFAULT_BATCH_SIZE_REF,
        help="Reference batch size B_ref; B★ = B_ref·N/N_ref.",
    )
    scale.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Reference learning rate α_ref; α★ = α_ref·(N/N_ref)^(-0.5).",
    )

    hp = parser.add_argument_group("fixed model / training hyperparameters")
    hp.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    hp.add_argument("--hidden-channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    hp.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    hp.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    hp.add_argument("--penalty-weight", type=float, default=DEFAULT_PENALTY_WEIGHT)
    hp.add_argument("--Lambda", type=float, default=DEFAULT_LAMBDA, dest="Lambda")
    hp.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    hp.add_argument("--hop", type=int, default=DEFAULT_HOP)
    hp.add_argument("--max-phy-loss", type=float, default=DEFAULT_MAX_PHY_LOSS)
    hp.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    hp.add_argument("--clip-value", type=float, default=DEFAULT_CLIP_VALUE)
    return parser.parse_args(argv)


def _fixed_hyperparameters(args: argparse.Namespace) -> dict:
    return {
        "decay": args.decay,
        "hidden_channels": args.hidden_channels,
        "heads": args.heads,
        "dropout": args.dropout,
        "penalty_weight": args.penalty_weight,
        "Lambda": args.Lambda,
        "alpha": args.alpha,
        "hop": args.hop,
        "max_phy_loss": args.max_phy_loss,
        "noise": args.noise,
        "clip_value": args.clip_value,
        "epochs": args.epochs,
        "folds": args.folds,
        "test_size": args.test_size,
        "n_ref": args.n_ref,
        "batch_size_ref": args.batch_size_ref,
        "learning_rate_ref": args.learning_rate,
        "n_trials": args.n_trials,
    }


def _scaling_centers(
    *,
    n_training: int,
    n_ref: int,
    batch_size_ref: int,
    learning_rate_ref: float,
) -> dict:
    ratio = n_training / n_ref
    batch_size_star = batch_size_ref * ratio
    learning_rate_star = learning_rate_ref * (ratio**-0.5)
    batch_low, batch_high = _batch_size_bounds(batch_size_star, n_training)
    return {
        "n_training": n_training,
        "n_ref": n_ref,
        "batch_size_ref": batch_size_ref,
        "learning_rate_ref": learning_rate_ref,
        "batch_size_star": batch_size_star,
        "learning_rate_star": learning_rate_star,
        "batch_size_low": batch_low,
        "batch_size_high": batch_high,
        "learning_rate_low": learning_rate_star / 10.0,
        "learning_rate_high": learning_rate_star * 10.0,
    }


def _batch_size_bounds(batch_size_star: float, n_training: int) -> tuple[int, int]:
    low = max(1, int(round(batch_size_star / 4.0)))
    high = min(n_training, max(1, int(round(batch_size_star * 4.0))))
    if high < low:
        center = max(1, min(n_training, int(round(batch_size_star))))
        low = high = center
    return low, high


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n")
    temp_path.replace(path)


@contextlib.contextmanager
def _working_directory(path: Path):
    """Keep the original trainer's default TensorBoard output inside a run."""
    original_directory = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_directory)


def _make_parser(
    data_dir: Path, device: torch.device, test_size: float
) -> DataParser:
    return DataParser(
        directory_path=str(data_dir),
        ref_re=REF_RE,
        ref_length=REF_LENGTH,
        root_chord_ref=ROOT_CHORD_REF,
        apex_x=0.0,
        apex_y=0.0,
        apex_z=0.0,
        test_size=test_size,
        device=str(device),
        use_robust_scaler=False,
        normalize_coordinates=True,
    )


class _Stub:
    """Stand-in for pickled SUAVE objects not needed to build panel graphs."""

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
            raise TypeError(f"{path.name} did not contain a list of MFData cases.")
        for case_index, case in enumerate(loaded):
            case.source_file = path.name
            case.case_index = case_index
        cases.extend(loaded)
        print(f"Loaded {len(loaded):4d} cases from {path.name}")

    for dataset_index, case in enumerate(cases):
        case.dataset_index = dataset_index
    print(f"Total cases from {directory}: {len(cases)}")
    return cases


def _build_raw_graphs(parser: DataParser, cases: list) -> list:
    graphs = []
    chunk_size = 20
    for start in range(0, len(cases), chunk_size):
        chunk = cases[start : start + chunk_size]
        assembled = parser.assemble_dataset(chunk)
        for index in range(len(chunk)):
            graphs.append(
                parser.prepare_dataset_with_standarization(
                    assembled["control_points"][index],
                    assembled["vlm_dcp"][index],
                    assembled["cfd_dcp"][index],
                    assembled["vlm_thickness"][index],
                    assembled["vlm_curvature_u"][index],
                    assembled["vlm_curvature_l"][index],
                    assembled["vlm_slope_chord_u"][index],
                    assembled["vlm_slope_chord_l"][index],
                    assembled["cfd_ff"][index],
                    assembled["aic_matrices"][index],
                    assembled["rhs_matrices"][index],
                    assembled["dcpsid_list"][index],
                    assembled["factor_list"][index],
                    assembled["chord_list"][index],
                    assembled["rnmax_list"][index],
                )
            )
        del assembled
    return graphs


def _case_metadata(cases: list, dataset_index: int) -> dict:
    case = cases[dataset_index]
    return {
        "dataset_index": int(case.dataset_index),
        "source_file": case.source_file,
        "case_index": int(case.case_index),
        "test_number": int(case.test_number),
        "mach": float(case.mach),
        "alpha": float(case.alpha),
        "Re": float(case.Re),
    }


def _model(args: argparse.Namespace, device: torch.device) -> GNN4:
    return GNN4(
        node_in_channels=11,
        edge_in_channels=1,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        dropout_rate=args.dropout,
        HEADS=args.heads,
        ALPHA=args.alpha,
        HOP=args.hop,
    ).to(device)


def _field_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    target_std = float(np.std(target))
    target_range = float(np.max(target) - np.min(target))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "nrmse_std": None if target_std == 0 else rmse / target_std,
        "nrmse_range": None if target_range == 0 else rmse / target_range,
        "n_panels": int(target.size),
        "target_std": target_std,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
    }


def _evaluate_graphs(
    model: GNN4,
    graphs: list,
    sources: list[str],
    device: torch.device,
) -> dict:
    if not graphs or len(graphs) != len(sources):
        raise ValueError("Evaluation graphs and source labels must be non-empty and aligned.")
    model.eval()
    predictions_by_source: dict[str, list[np.ndarray]] = {}
    targets_by_source: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for graph, source in zip(graphs, sources):
            graph = graph.to(device)
            prediction = model(graph).detach().cpu().numpy().reshape(-1)
            target = graph.y.detach().cpu().numpy().reshape(-1)
            predictions_by_source.setdefault(source, []).append(prediction)
            targets_by_source.setdefault(source, []).append(target)

    all_predictions = np.concatenate(
        [array for values in predictions_by_source.values() for array in values]
    )
    all_targets = np.concatenate(
        [array for values in targets_by_source.values() for array in values]
    )
    summary = _field_metrics(all_predictions, all_targets)
    summary["n_cases"] = len(graphs)
    summary["by_source"] = {
        source: {
            **_field_metrics(
                np.concatenate(predictions_by_source[source]),
                np.concatenate(targets_by_source[source]),
            ),
            "n_cases": len(predictions_by_source[source]),
        }
        for source in predictions_by_source
    }
    return summary


def _write_study_summary(study: optuna.Study, path: Path, centers: dict) -> None:
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best = min(completed, key=lambda trial: trial.value) if completed else None
    _write_json(
        path,
        {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "study": study.study_name,
            "direction": "minimize",
            "objective": "training_geometry_holdout.rmse",
            "scaling_centers": centers,
            "n_trials": len(study.trials),
            "best_trial": None
            if best is None
            else {
                "number": best.number,
                "value": best.value,
                "params": best.params,
            },
            "trials": trials,
        },
    )


def _write_aggregate(output_dir: Path, result: dict) -> None:
    summary_path = output_dir / "scaling_results.json"
    previous_results = []
    if summary_path.exists():
        previous_results = json.loads(summary_path.read_text()).get("results", [])

    key = (int(result["seed"]), round(float(result["fraction"]) * 100))
    results = [
        item
        for item in previous_results
        if (int(item["seed"]), round(float(item["fraction"]) * 100)) != key
    ]
    results.append(result)
    results.sort(key=lambda item: (int(item["seed"]), float(item["fraction"])))
    _write_json(
        summary_path,
        {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "subset_sampling": "Nested random subsets of dataset/train_set per seed.",
            "validation_split": "70/30 training/validation within each nested subset.",
            "objective": "training_geometry_holdout.rmse",
            "optuna": (
                "Per fraction, search batch_size and learning_rate around "
                "B★=B_ref·N/N_ref and α★=α_ref·(N/N_ref)^(-0.5)."
            ),
            "test_set": "dataset/new_geom (reported only; not used for selection).",
            "test_set_used_for_selection": False,
            "results": results,
        },
    )

    csv_path = output_dir / "scaling_results.csv"
    temp_path = csv_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in item.items()
            }
            for item in results
        )
    temp_path.replace(csv_path)


def _write_train_validation_split(
    *,
    output_dir: Path,
    fraction_dir: Path,
    seed: int,
    fraction: float,
    fit_indices: list[int],
    validation_indices: list[int],
    cases: list,
    folds: int,
) -> None:
    rows = [
        {
            "seed": seed,
            "fraction": fraction,
            "split": split,
            "graph_index": graph_index,
            **_case_metadata(cases, dataset_index),
        }
        for split, indices in (
            ("training", fit_indices),
            ("validation", validation_indices),
        )
        for graph_index, dataset_index in enumerate(indices)
    ]
    cv_rows = []
    splitter = KFold(n_splits=folds, shuffle=True, random_state=CV_RANDOM_STATE)
    for fold, (training_positions, validation_positions) in enumerate(
        splitter.split(fit_indices), start=1
    ):
        for role, positions in (
            ("training", training_positions),
            ("validation", validation_positions),
        ):
            for position in positions:
                dataset_index = fit_indices[int(position)]
                cv_rows.append(
                    {
                        "seed": seed,
                        "fraction": fraction,
                        "fold": fold,
                        "role": role,
                        **_case_metadata(cases, dataset_index),
                    }
                )

    split_path = output_dir / "train_validation_split.csv"
    previous_rows = []
    if split_path.exists():
        with split_path.open(newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
    rows.extend(
        row
        for row in previous_rows
        if not (
            int(row["seed"]) == seed
            and round(float(row["fraction"]) * 100) == round(fraction * 100)
        )
    )
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            float(row["fraction"]),
            int(row["dataset_index"]),
            row["split"],
        )
    )

    fraction_split_path = fraction_dir / "train_validation_split.csv"
    for path in (split_path, fraction_split_path):
        rows_to_write = rows
        if path == fraction_split_path:
            rows_to_write = [
                row
                for row in rows
                if int(row["seed"]) == seed
                and round(float(row["fraction"]) * 100) == round(fraction * 100)
            ]
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_to_write)
        temp_path.replace(path)

    cv_rows.sort(
        key=lambda row: (
            int(row["fold"]),
            row["role"],
            int(row["dataset_index"]),
        )
    )
    with (fraction_dir / "cv_fold_split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CV_SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(cv_rows)


def _run_fraction(
    *,
    args: argparse.Namespace,
    seed: int,
    fraction: float,
    cases: list,
    raw_graphs: list,
    test_cases: list,
    raw_test_graphs: list,
    parser: DataParser,
    device: torch.device,
    output_dir: Path,
) -> dict:
    percent = round(fraction * 100)
    n_samples = max(1, round(len(cases) * fraction))
    selected_indices = np.random.default_rng(seed).permutation(len(cases))[:n_samples]
    selection_split_seed = seed + percent
    fit_indices, validation_indices = train_test_split(
        selected_indices.tolist(),
        test_size=args.test_size,
        random_state=selection_split_seed,
    )
    if len(fit_indices) < args.folds:
        raise ValueError(
            f"Seed {seed}, {percent}% fraction has only {len(fit_indices)} fit cases; "
            f"at least {args.folds} are required for cross-validation."
        )

    fraction_dir = output_dir / f"seed_{seed}" / f"fraction_{percent:02d}"
    fraction_dir.mkdir(parents=True, exist_ok=True)
    run_dir = fraction_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = fraction_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    parser.nn_training_set = copy.deepcopy([raw_graphs[i] for i in fit_indices])
    parser.nn_test_set = copy.deepcopy(
        [raw_graphs[i] for i in validation_indices] + raw_test_graphs
    )
    parser.scaler = None
    training_graphs, all_evaluation_graphs = parser.normalize_data()
    n_validation = len(validation_indices)
    validation_graphs = all_evaluation_graphs[:n_validation]
    test_graphs = all_evaluation_graphs[n_validation:]
    validation_sources = [cases[index].source_file for index in validation_indices]
    test_sources = [case.source_file for case in test_cases]

    centers = _scaling_centers(
        n_training=len(fit_indices),
        n_ref=args.n_ref,
        batch_size_ref=args.batch_size_ref,
        learning_rate_ref=args.learning_rate,
    )

    split = {
        "seed": seed,
        "fraction": fraction,
        "test_size": args.test_size,
        "selection_split_seed": selection_split_seed,
        "cv_split_seed": CV_RANDOM_STATE,
        "n_folds": args.folds,
        "n_selected": len(selected_indices),
        "n_training": len(fit_indices),
        "n_validation": len(validation_indices),
        "scaling_centers": centers,
        "selected_cases": [_case_metadata(cases, int(i)) for i in selected_indices],
        "training_cases": [_case_metadata(cases, i) for i in fit_indices],
        "validation_cases": [_case_metadata(cases, i) for i in validation_indices],
        "test_case_count": len(test_cases),
        "test_geometry_source_count": len({case.source_file for case in test_cases}),
        "test_geometry_cases": [
            _case_metadata(test_cases, index) for index in range(len(test_cases))
        ],
    }
    _write_json(fraction_dir / "split.json", split)
    _write_train_validation_split(
        output_dir=output_dir,
        fraction_dir=fraction_dir,
        seed=seed,
        fraction=fraction,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        cases=cases,
        folds=args.folds,
    )

    print(
        f"Seed {seed}, {percent}% scaling centers: "
        f"N={centers['n_training']}, "
        f"B★={centers['batch_size_star']:.4g} "
        f"[{centers['batch_size_low']}, {centers['batch_size_high']}], "
        f"α★={centers['learning_rate_star']:.6g} "
        f"[{centers['learning_rate_low']:.6g}, {centers['learning_rate_high']:.6g}]"
    )

    study = optuna.create_study(
        study_name=f"mf_scaling_seed_{seed}_fraction_{percent:02d}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=CV_RANDOM_STATE + percent),
        storage=f"sqlite:///{fraction_dir / 'study.db'}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        _set_seed(CV_RANDOM_STATE + trial.number)
        batch_size = trial.suggest_int(
            "batch_size",
            centers["batch_size_low"],
            centers["batch_size_high"],
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            centers["learning_rate_low"],
            centers["learning_rate_high"],
            log=True,
        )
        model = _model(args, device)
        with _working_directory(fraction_dir):
            fold_losses, trained_model, mean_validation_loss = train_model_k_fold(
                model,
                training_graphs,
                k=args.folds,
                learning_rate=learning_rate,
                epochs=args.epochs,
                clip_value=args.clip_value,
                noise_level=args.noise,
                penalty_weight=args.penalty_weight,
                device=str(device),
                Lambda=args.Lambda,
                decay=args.decay,
                max_phy_loss=args.max_phy_loss,
                batch_size=batch_size,
            )
        holdout_metrics = _evaluate_graphs(
            trained_model, validation_graphs, validation_sources, device
        )
        trial.set_user_attr("fold_validation_losses", [float(v) for v in fold_losses])
        trial.set_user_attr("cv_mean_validation_loss", float(mean_validation_loss))
        trial.set_user_attr("holdout_rmse", holdout_metrics["rmse"])
        trial.set_user_attr("holdout_mse", holdout_metrics["mse"])
        checkpoint_path = trials_dir / f"trial_{trial.number:03d}.pt"
        torch.save(trained_model.state_dict(), checkpoint_path)
        del model, trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"Seed {seed}, {percent}% trial {trial.number}: "
            f"batch_size={batch_size}, lr={learning_rate:.6g}, "
            f"validation RMSE {holdout_metrics['rmse']:.6g}"
        )
        return float(holdout_metrics["rmse"])

    remaining_trials = max(0, args.n_trials - len(study.trials))
    if remaining_trials:
        print(
            f"Seed {seed}, {percent}%: running {remaining_trials} "
            f"of {args.n_trials} Optuna trials"
        )
        study.optimize(objective, n_trials=remaining_trials)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(f"No completed Optuna trial for seed {seed}, {percent}%.")
    best_trial = min(completed_trials, key=lambda trial: trial.value)
    candidate_path = trials_dir / f"trial_{best_trial.number:03d}.pt"
    report_path = run_dir / "report.json"
    model_path = run_dir / "model.pth"

    if candidate_path.is_file():
        shutil.copy2(candidate_path, model_path)
    elif (
        not model_path.is_file()
        or not report_path.is_file()
        or int(json.loads(report_path.read_text()).get("best_trial", -1))
        != best_trial.number
    ):
        raise FileNotFoundError(
            f"Best trial checkpoint is missing for seed {seed}, {percent}%."
        )

    model = _model(args, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    validation_metrics = _evaluate_graphs(
        model, validation_graphs, validation_sources, device
    )
    test_metrics = _evaluate_graphs(model, test_graphs, test_sources, device)
    del model

    report = {
        "seed": seed,
        "fraction": fraction,
        "selection_metric": "training_geometry_holdout.rmse",
        "test_geometry_used_for_selection": False,
        "n_selected": len(selected_indices),
        "n_training": len(fit_indices),
        "n_validation": len(validation_indices),
        "n_test_cases": len(test_graphs),
        "selection_split_seed": selection_split_seed,
        "cv_split_seed": CV_RANDOM_STATE,
        "scaling_centers": centers,
        "best_trial": best_trial.number,
        "validation_rmse": float(best_trial.value),
        "cv_mean_validation_loss": best_trial.user_attrs.get(
            "cv_mean_validation_loss"
        ),
        "fold_validation_losses": best_trial.user_attrs.get("fold_validation_losses"),
        "validation_geometry": validation_metrics,
        "final_test_geometry": test_metrics,
        "test_rmse": float(test_metrics["rmse"]),
        "best_params": best_trial.params,
        "hyperparameters": {
            **_fixed_hyperparameters(args),
            **best_trial.params,
        },
        "checkpoint": str(model_path),
        "scaler": str(run_dir / "scaler.pkl"),
        "split": str(fraction_dir / "split.json"),
    }
    _write_json(report_path, report)
    parser.save_scaler(str(run_dir / "scaler.pkl"))
    _write_study_summary(study, fraction_dir / "study_summary.json", centers)
    _write_aggregate(output_dir, report)

    for trial in study.trials:
        (trials_dir / f"trial_{trial.number:03d}.pt").unlink(missing_ok=True)

    del training_graphs, validation_graphs, test_graphs, all_evaluation_graphs
    parser.nn_training_set = None
    parser.nn_test_set = None
    parser.scaler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"Seed {seed}, {percent}% complete: "
        f"best trial {best_trial.number} "
        f"params={best_trial.params}, "
        f"validation RMSE {report['validation_rmse']:.6g}, "
        f"new-geometry test RMSE {report['test_rmse']:.6g}"
    )
    return report


def run(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("--epochs must be positive; --folds must be at least 2.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique seed values.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seed values must be non-negative.")
    if not args.fractions or len(set(args.fractions)) != len(args.fractions):
        raise ValueError("--fractions must contain unique fractions.")
    if args.learning_rate <= 0 or args.decay < 0:
        raise ValueError("--learning-rate must be positive; --decay must be non-negative.")
    if args.n_ref < 1 or args.batch_size_ref < 1:
        raise ValueError("--n-ref and --batch-size-ref must be positive integers.")
    if args.hidden_channels < 1 or args.heads < 1 or args.hop < 1:
        raise ValueError("--hidden-channels, --heads, and --hop must be positive integers.")

    train_dir = Path(args.train_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output_dir / "train_log.txt").open("a", buffering=1) as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            print(
                f"\nMF scaling run started "
                f"{datetime.now().isoformat(timespec='seconds')}"
            )
            print(f"Device: {device}")
            print(f"Training geometries: {train_dir}")
            print(f"Final test geometries: {test_dir}")
            print(f"Output: {output_dir}")
            print(
                "Fixed hyperparameters / scaling refs: "
                f"{json.dumps(_fixed_hyperparameters(args), sort_keys=True)}"
            )

            parser = _make_parser(train_dir, device, args.test_size)
            cases = _load_cases(train_dir, validation=False)
            test_cases = _load_cases(test_dir, validation=True)
            if not cases:
                raise ValueError(f"No training cases found in {train_dir}.")
            if not test_cases:
                raise ValueError(f"No test-geometry cases found in {test_dir}.")

            print("\nBuilding raw training graphs")
            raw_graphs = _build_raw_graphs(parser, cases)
            if len(raw_graphs) != len(cases):
                raise RuntimeError(
                    "Training graph construction did not produce one graph per case."
                )
            print("\nBuilding raw final-test geometry graphs")
            raw_test_graphs = _build_raw_graphs(parser, test_cases)
            if len(raw_test_graphs) != len(test_cases):
                raise RuntimeError(
                    "Test graph construction did not produce one graph per case."
                )

            for seed in args.seeds:
                for fraction in args.fractions:
                    _run_fraction(
                        args=args,
                        seed=seed,
                        fraction=fraction,
                        cases=cases,
                        raw_graphs=raw_graphs,
                        test_cases=test_cases,
                        raw_test_graphs=raw_test_graphs,
                        parser=parser,
                        device=device,
                        output_dir=output_dir,
                    )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        traceback.print_exc()
        raise
