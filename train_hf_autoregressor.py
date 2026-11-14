"""Train VortexNet as a high-fidelity autoregressor and report field error.

The panel graph is the same one built by the delta-wing trainer:
VortexNetUtils.assemble_dataset and prepare_dataset_with_standarization
produce 11 node features and one edge feature (distance to the four nearest
neighbors). The first node feature is the pressure coefficient. Here that
column is the high-fidelity CFD ΔCp instead of the vortex-lattice ΔCp, and
the training target is the same HF field.

Two evaluations are written to this folder's output directory:

1. Training-geometry holdout. Cases in dataset/train_set are randomly split
   80/20. The model is trained on the 80% and the error is measured on the
   held-out 20%, which still comes from the training geometries.
2. Test-geometry set. The same checkpoint is evaluated on dataset/new_geom
   (sweep 60 and 70 validation wings). Those geometries are not used for
   training or for fitting the scaler.

Run from the repository root:

    python VortexNet/hf_autoregressor/train_hf_autoregressor.py

Hyperparameter search for the same model lives in
search_hp_hf_autoregressor.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from collections import Counter
from datetime import datetime
from pathlib import Path


import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[0]
PACKAGE_DIR = Path(__file__).resolve().parent
# Pickled cases reference a top-level MFData module, which lives in VortexNet/.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VortexNet"))
# The bundled SUAVE install is not needed for graph construction and fails to
# import in this environment. Stub those classes while unpickling instead.
sys.path = [p for p in sys.path if "SUAVE" not in p]

from VortexNet.model import GNN4
from VortexNet.training import PenalizedSmoothL1Loss
from VortexNet.VortexNetUtils import VortexNetUtils


_ALLOWED_MODULES = (
    "MFData",
    "numpy",
    "collections",
    "copyreg",
    "builtins",
    "__builtin__",
    "scipy",
)


class _Stub:
    """Stand-in for pickled SUAVE objects such as the VLM vortex distribution."""

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


class _NoSuaveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_ALLOWED_MODULES) or module in _ALLOWED_MODULES:
            return super().find_class(module, name)

        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            modname = ".".join(parts[:i])
            if modname not in sys.modules:
                mod = types.ModuleType(modname)
                sys.modules[modname] = mod
                if i > 1:
                    setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
        mod = sys.modules[module]
        if not hasattr(mod, name):
            setattr(mod, name, type(name, (_Stub,), {"__module__": module}))
        return getattr(mod, name)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return _NoSuaveUnpickler(handle).load()


def _case_files(directory: Path, validation: bool) -> list[Path]:
    files = sorted(directory.glob("pklData_*.pkl"))
    if validation:
        return [path for path in files if "validation" in path.name]
    return [path for path in files if "validation" not in path.name]


def load_cases(
    directory: Path,
    validation: bool,
    max_cases: int | None = None,
    file_contains: str | None = None,
) -> list:
    """Load MFData cases from a dataset directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    files = _case_files(directory, validation=validation)
    if file_contains:
        files = [path for path in files if file_contains in path.name]
    
    if not files:
        kind = "validation" if validation else "training"
        suffix = f" containing {file_contains!r}" if file_contains else ""
        raise FileNotFoundError(
            f"No {kind} pklData_*.pkl files{suffix} in {directory}."
        )

    cases = []
    for path in files:
        loaded = _load_pickle(path)
        if not isinstance(loaded, list):
            raise TypeError(f"{path.name} did not contain a list of MFData cases.")
        for case_index, case in enumerate(loaded):
            case.source_file = path.name
            case.case_index = case_index
        cases.extend(loaded)
        print(f"Loaded {len(loaded):4d} cases from {path.name}")
        if max_cases is not None and len(cases) >= max_cases:
            cases = cases[:max_cases]
            print(f"Stopped after {max_cases} cases (--max-cases).")
            break
    print(f"Total cases from {directory}: {len(cases)}")
    return cases


# Same widths as the delta-wing GNN4: 11 node features, distance on each edge.
NODE_IN_CHANNELS = 11
EDGE_IN_CHANNELS = 1


def _as_hf_autoregressor(data: Data) -> Data:
    """Keep the 11-feature panel graph and put HF ΔCp in the pressure column.

    prepare_dataset_with_standarization stores vortex-lattice ΔCp in column 0
    and the CFD field in ``y``. The other columns stay as that function built
    them. VLM system matrices are dropped because this trainer does not use
    the physics residual.
    """
    target = data.y.detach().cpu().contiguous().view(-1, 1)
    features = data.x.detach().cpu().contiguous().clone()
    if features.shape[1] != NODE_IN_CHANNELS:
        raise ValueError(
            f"Expected {NODE_IN_CHANNELS} node features, got {features.shape[1]}."
        )
    if target.shape[0] != features.shape[0]:
        raise ValueError(
            f"HF target length {target.shape[0]} does not match {features.shape[0]} panels."
        )
    if not torch.isfinite(target).all() or not torch.isfinite(features).all():
        raise ValueError("Panel graph contains non-finite values.")
    features[:, 0:1] = target
    return Data(
        x=features,
        edge_index=data.edge_index.detach().cpu().contiguous(),
        edge_attr=data.edge_attr.detach().cpu().contiguous(),
        y=target.clone(),
    )


def build_hf_graphs(cases: list, ref_re: float) -> tuple[list[Data], list[str]]:
    """Build graphs the same way as the delta-wing training script."""
    graphs = []
    sources = []
    # Assemble in chunks so the dense VLM influence matrices are not all resident.
    chunk_size = 20
    for start in range(0, len(cases), chunk_size):
        chunk = cases[start:start + chunk_size]
        assembled = VortexNetUtils.assemble_dataset(chunk, ref_re)
        rows = zip(
            assembled["control_points"],
            assembled["vlm_dcp"],
            assembled["cfd_dcp"],
            assembled["vlm_thickness"],
            assembled["vlm_curvature_u"],
            assembled["vlm_curvature_l"],
            assembled["vlm_slope_chord_u"],
            assembled["vlm_slope_chord_l"],
            assembled["cfd_ff"],
            assembled["aic_matrices"],
            assembled["rhs_matrices"],
            assembled["dcpsid_list"],
            assembled["factor_list"],
            assembled["chord_list"],
            assembled["rnmax_list"],
        )
        for case, row in zip(chunk, rows):
            (
                cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l,
                slope_u, slope_l, ff, aic_matrix, rhs_matrix, dcpsid,
                factor, chord, rnmax,
            ) = row
            graph = VortexNetUtils.prepare_dataset_with_standarization(
                cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l,
                slope_u, slope_l, ff, aic_matrix, rhs_matrix, dcpsid,
                factor, chord, rnmax,
            )
            graphs.append(_as_hf_autoregressor(graph))
            sources.append(case.source_file)
        del assembled
        print(f"  built {len(graphs)}/{len(cases)} graphs")
    if len(graphs) != len(cases):
        raise RuntimeError("Graph build did not return one graph per case.")
    if graphs and graphs[0].num_edge_features != EDGE_IN_CHANNELS:
        raise RuntimeError(
            f"Expected {EDGE_IN_CHANNELS} edge feature, got {graphs[0].num_edge_features}."
        )
    return graphs, sources


def make_model(args, device: torch.device) -> GNN4:
    model = GNN4(
        node_in_channels=NODE_IN_CHANNELS,
        edge_in_channels=EDGE_IN_CHANNELS,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        dropout_rate=args.dropout,
        HEADS=args.heads,
        ALPHA=args.alpha,
        HOP=args.hop,
    )
    return model.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, noise_level, clip_value) -> float:
    model.train()
    loss_sum = 0.0
    node_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        # Match train_model_k_fold: leave the pressure column unchanged.
        noisy_x = batch.x.clone()
        if noise_level > 0 and noisy_x.shape[1] > 1:
            noisy_x[:, 1:] = noisy_x[:, 1:] + noise_level * torch.randn_like(noisy_x[:, 1:])
        batch.x = noisy_x
        prediction = model(batch)
        loss = criterion(prediction, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        nodes = batch.y.numel()
        loss_sum += loss.item() * nodes
        node_count += nodes
    return loss_sum / max(node_count, 1)


@torch.no_grad()
def predict_normalized(model, graphs: list[Data], device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).detach().cpu().numpy().reshape(-1))
        targets.append(batch.y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def field_metrics(pred_raw: np.ndarray, true_raw: np.ndarray, pred_norm: np.ndarray, true_norm: np.ndarray) -> dict:
    error = pred_raw - true_raw
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    std = float(np.std(true_raw))
    span = float(np.max(true_raw) - np.min(true_raw))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse_std": None if std == 0 else rmse / std,
        "nrmse_range": None if span == 0 else rmse / span,
        "mse_normalized": float(np.mean((pred_norm - true_norm) ** 2)),
        "n_panels": int(true_raw.shape[0]),
        "target_std": std,
        "target_min": float(np.min(true_raw)),
        "target_max": float(np.max(true_raw)),
    }


def metrics_by_source(pred_raw, true_raw, pred_norm, true_norm, sources, counts) -> dict:
    grouped: dict[str, dict] = {}
    offset = 0
    buckets: dict[str, list[tuple[np.ndarray, ...]]] = {}
    case_counts: Counter = Counter()
    for source, count in zip(sources, counts):
        sl = slice(offset, offset + count)
        buckets.setdefault(source, []).append(
            (pred_raw[sl], true_raw[sl], pred_norm[sl], true_norm[sl])
        )
        case_counts[source] += 1
        offset += count
    if offset != len(pred_raw):
        raise RuntimeError(
            f"Panel count {offset} does not match predictions {len(pred_raw)}."
        )
    for source, parts in buckets.items():
        grouped[source] = field_metrics(
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
            np.concatenate([part[3] for part in parts]),
        )
        grouped[source]["n_cases"] = int(case_counts[source])
    return grouped


def evaluate_split(model, graphs, sources, scaler, device, batch_size) -> dict:
    pred_norm, true_norm = predict_normalized(model, graphs, device, batch_size)
    if scaler is None:
        pred_raw = pred_norm
        true_raw = true_norm
    else:
        pred_raw = scaler.inverse_transform(pred_norm.reshape(-1, 1)).reshape(-1)
        true_raw = scaler.inverse_transform(true_norm.reshape(-1, 1)).reshape(-1)
    counts = [int(graph.num_nodes) for graph in graphs]
    summary = field_metrics(pred_raw, true_raw, pred_norm, true_norm)
    summary["n_cases"] = len(graphs)
    summary["by_source"] = metrics_by_source(
        pred_raw, true_raw, pred_norm, true_norm, sources, counts
    )
    return summary


class _Tee:
    """Mirror writes to the real console and a log file.

    Delegates attributes such as ``isatty`` so libraries that inspect the
    console stream keep working after stdout is replaced.
    """

    def __init__(self, log_file, console):
        self.log_file = log_file
        self.console = console

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def cpu_state_dict(state):
    # Clone after moving to CPU. state_dict() shares storage with the live
    # parameters, and Tensor.cpu() is a no-op when the tensor is already there.
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _format_metrics(summary: dict) -> str:
    return (
        f"RMSE {summary['rmse']:.6f}   "
        f"NRMSE(std) {summary['nrmse_std']:.6f}   "
        f"NRMSE(range) {summary['nrmse_range']:.6f}   "
        f"MAE {summary['mae']:.6f}   "
        f"MSE {summary['mse']:.6f}   "
        f"cases {summary['n_cases']}"
    )


ERROR_DEFINITION = {
    "rmse": "sqrt(mean((prediction - HF ΔCp)^2))",
    "nrmse_std": "rmse / std(HF ΔCp)",
    "nrmse_range": "rmse / (max(HF ΔCp) - min(HF ΔCp))",
    "mae": "mean(abs(prediction - HF ΔCp))",
    "mse": "mean((prediction - HF ΔCp)^2)",
    "mse_normalized": "mean squared error of the network output against the training target",
}


def _plain_number(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _split_record(case, split: str, graph_index: int) -> dict:
    return {
        "split": split,
        "dataset_index": int(case.dataset_index),
        "graph_index": graph_index,
        "source_file": case.source_file,
        "case_index": int(case.case_index),
        "test_number": _plain_number(case.test_number),
        "mach": _plain_number(case.mach),
        "alpha": _plain_number(case.alpha),
        "Re": _plain_number(case.Re),
    }


def train_set_split_manifest(args, fit_cases, holdout_cases) -> dict:
    """Identity of every dataset/train_set graph in the random fit/holdout split."""
    training = [
        _split_record(case, "training", graph_index)
        for graph_index, case in enumerate(fit_cases)
    ]
    test = [
        _split_record(case, "test", graph_index)
        for graph_index, case in enumerate(holdout_cases)
    ]
    training.sort(key=lambda record: record["dataset_index"])
    test.sort(key=lambda record: record["dataset_index"])
    return {
        "description": (
            "Random split of dataset/train_set. "
            "training is the fit split. test is the held-out portion of those same geometries. "
            "dataset/new_geom is not listed here."
        ),
        "seed": args.seed,
        "test_size": args.test_size,
        "n_training": len(training),
        "n_test": len(test),
        "training": training,
        "test": test,
    }


def write_train_set_split(out_dir: Path, manifest: dict) -> tuple[Path, Path]:
    """Write the training-set assignment as JSON and as a plain-text table."""
    out_dir = Path(out_dir)
    json_path = out_dir / "train_set_split.json"
    text_path = out_dir / "train_set_split.txt"
    with json_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    columns = (
        "dataset_index", "graph_index", "source_file", "case_index",
        "test_number", "mach", "alpha", "Re",
    )
    lines = [
        manifest["description"],
        f"seed: {manifest['seed']}",
        f"test_size: {manifest['test_size']}",
        f"training: {manifest['n_training']} graphs",
        f"test: {manifest['n_test']} graphs",
        "",
    ]
    for split_name in ("training", "test"):
        lines.append(split_name)
        lines.append("\t".join(columns))
        for record in manifest[split_name]:
            lines.append("\t".join(str(record[column]) for column in columns))
        lines.append("")
    text_path.write_text("\n".join(lines))
    return json_path, text_path


def prepare_hf_data(args):
    """Load both geometry sets and split the training geometries 80/20.

    Graphs use the delta-wing 11-feature layout. Column 0 and ``y`` are the
    raw HF CFD ΔCp. No extra StandardScaler is applied.
    """
    test_files = _case_files(Path(args.test_dir), validation=True)
    if not test_files:
        raise FileNotFoundError(
            f"No test-geometry validation pickles in {args.test_dir}."
        )
    print(f"Test-geometry files: {len(test_files)}")


    train_file_contains = getattr(args, "train_file_contains", None)
    print("\nLoading training geometries")
    if train_file_contains:
        print(f"Training files restricted to names containing {train_file_contains!r}")
    train_cases = load_cases(
        Path(args.train_dir),
        validation=False,
        max_cases=args.max_cases,
        file_contains=train_file_contains,
    )

    for dataset_index, case in enumerate(train_cases):
        case.dataset_index = dataset_index
    fit_cases, holdout_cases = VortexNetUtils.split_mfdata(
        train_cases, test_size=args.test_size, random_seeds=args.seed
    )

    print("\nBuilding training-split graphs")
    fit_graphs, fit_sources = build_hf_graphs(fit_cases, args.ref_re)
    print("Building training-geometry holdout graphs")
    holdout_graphs, holdout_sources = build_hf_graphs(holdout_cases, args.ref_re)
    print(
        f"Nodes per graph: {fit_graphs[0].num_nodes}, "
        f"edges per graph: {fit_graphs[0].num_edges}, "
        f"node features: {fit_graphs[0].num_features}, "
        f"edge features: {fit_graphs[0].num_edge_features}"
    )

    print("\nLoading test geometries")
    test_cases = load_cases(Path(args.test_dir), validation=True, max_cases=None)
    print("Building test-geometry graphs")
    test_graphs, test_sources = build_hf_graphs(test_cases, args.ref_re)

    return {
        "scaler": None,
        "fit_graphs": fit_graphs,
        "fit_sources": fit_sources,
        "holdout_graphs": holdout_graphs,
        "holdout_sources": holdout_sources,
        "test_graphs": test_graphs,
        "test_sources": test_sources,
        "train_set_split": train_set_split_manifest(args, fit_cases, holdout_cases),
    }


def model_kwargs_from_spec(spec) -> dict:
    return {
        "node_in_channels": NODE_IN_CHANNELS,
        "edge_in_channels": EDGE_IN_CHANNELS,
        "hidden_channels": spec.hidden_channels,
        "out_channels": 1,
        "dropout_rate": spec.dropout,
        "HEADS": spec.heads,
        "ALPHA": spec.alpha,
        "HOP": spec.hop,
    }


def fit_hf_model(spec, prepared, device, trial=None, partial=None):
    """Train GNN4 on the 80% split.

    The kept checkpoint is the one with the lowest training SmoothL1. When
    ``trial`` is set, each epoch reports holdout RMSE to Optuna and may raise
    ``TrialPruned``. ``partial`` then receives the best checkpoint so far.
    """
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    model = make_model(spec, device)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"GNN4 parameters: {n_params}")
    criterion = PenalizedSmoothL1Loss(penalty_weight=spec.penalty_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=spec.scheduler_patience, factor=0.5
    )
    loader = DataLoader(prepared["fit_graphs"], batch_size=spec.batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = cpu_state_dict(model.state_dict())
    stall = 0
    history = []
    kwargs = model_kwargs_from_spec(spec)

    def snapshot():
        return {
            "model": model,
            "best_state": best_state,
            "best_train_smoothl1": best_loss,
            "n_parameters": n_params,
            "history": history,
            "model_kwargs": kwargs,
        }

    for epoch in range(spec.epochs):
        train_loss = train_one_epoch(
            model, loader, optimizer, criterion, device,
            spec.noise, spec.clip_value,
        )
        scheduler.step(train_loss)
        lr = optimizer.param_groups[0]["lr"]
        holdout = evaluate_split(
            model, prepared["holdout_graphs"], prepared["holdout_sources"],
            prepared["scaler"], device, spec.batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_smoothl1": train_loss,
                "learning_rate": lr,
                "holdout_rmse": holdout["rmse"],
                "holdout_nrmse_std": holdout["nrmse_std"],
                "holdout_mse": holdout["mse"],
            }
        )
        print(
            f"Epoch {epoch:3d}  train_smoothl1 {train_loss:.6f}  lr {lr:.3e}  "
            f"holdout {_format_metrics(holdout)}"
        )
        if not np.isfinite(train_loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {train_loss}")
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = cpu_state_dict(model.state_dict())
            stall = 0
        else:
            stall += 1

        if trial is not None:
            trial.report(holdout["rmse"], step=epoch)
            if trial.should_prune():
                if partial is not None:
                    partial.update(snapshot())
                import optuna
                print(f"Pruned at epoch {epoch}. Best train SmoothL1 {best_loss:.6f}")
                raise optuna.TrialPruned()

        if stall >= spec.patience:
            print(f"Early stopping at epoch {epoch}. Best train SmoothL1 {best_loss:.6f}")
            break

    model.load_state_dict(best_state)
    return snapshot()


def evaluate_prepared(model, prepared, device, batch_size):
    """Score the fit split, the training-geometry holdout, and the test geometries."""
    scaler = prepared["scaler"]
    fit_metrics = evaluate_split(
        model, prepared["fit_graphs"], prepared["fit_sources"], scaler, device, batch_size
    )
    holdout_metrics = evaluate_split(
        model, prepared["holdout_graphs"], prepared["holdout_sources"],
        scaler, device, batch_size,
    )
    test_metrics = evaluate_split(
        model, prepared["test_graphs"], prepared["test_sources"], scaler, device, batch_size
    )
    return fit_metrics, holdout_metrics, test_metrics


def build_report(spec, prepared, trained, fit_metrics, holdout_metrics, test_metrics) -> dict:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "GNN4 autoregressor on the delta-wing 11-feature panel graph. "
            "Node feature 0 and the target are raw HF CFD ΔCp. "
            "Errors below are on that same ΔCp."
        ),
        "error_definition": ERROR_DEFINITION,
        "checkpoint_selected_by": "training SmoothL1 on the 80% split",
        "hyperparameters": {
            "hidden_channels": spec.hidden_channels,
            "heads": spec.heads,
            "hop": spec.hop,
            "alpha": spec.alpha,
            "dropout": spec.dropout,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "noise": spec.noise,
            "clip_value": spec.clip_value,
            "batch_size": spec.batch_size,
            "epochs_requested": spec.epochs,
            "patience": spec.patience,
            "scheduler_patience": spec.scheduler_patience,
            "penalty_weight": spec.penalty_weight,
            "seed": spec.seed,
            "test_size": spec.test_size,
            "n_parameters": trained["n_parameters"],
        },
        "split": {
            "n_fit": len(prepared["fit_graphs"]),
            "n_holdout": len(prepared["holdout_graphs"]),
            "n_test_geometry": len(prepared["test_graphs"]),
            "fit_files": dict(Counter(prepared["fit_sources"])),
            "holdout_files": dict(Counter(prepared["holdout_sources"])),
            "test_files": dict(Counter(prepared["test_sources"])),
        },
        "training_report": {
            "fit": fit_metrics,
            "holdout": holdout_metrics,
            "history": trained["history"],
        },
        "test_report": test_metrics,
        "training_fit": fit_metrics,
        "training_geometry_holdout": holdout_metrics,
        "test_geometry": test_metrics,
        "history": trained["history"],
    }


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.txt"
    log_file = log_path.open("w")
    sys.stdout = _Tee(log_file, sys.__stdout__)
    sys.stderr = _Tee(log_file, sys.__stderr__)

    device = torch.device(args.device)
    print(f"HF autoregressor training  {datetime.now().isoformat(timespec='seconds')}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(
        "GNN4 uses 11 node features and 1 edge feature, as in the delta-wing trainer. "
        "Feature 0 and the target are raw HF ΔCp."
    )
    prepared = prepare_hf_data(args)
    split_json, split_text = write_train_set_split(out_dir, prepared["train_set_split"])
    print(f"Wrote {split_json}")
    print(f"Wrote {split_text}")

    trained = fit_hf_model(args, prepared, device)
    print("\nEvaluating the checkpoint with the lowest training loss")
    fit_metrics, holdout_metrics, test_metrics = evaluate_prepared(
        trained["model"], prepared, device, args.batch_size
    )
    checkpoint = {
        "model_state": trained["best_state"],
        "model_kwargs": trained["model_kwargs"],
        "best_train_smoothl1": trained["best_train_smoothl1"],
    }
    torch.save(checkpoint, out_dir / "model.pt")
    report = build_report(
        args, prepared, trained, fit_metrics, holdout_metrics, test_metrics
    )
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 72)
    print("1. Training-geometry holdout (random 20% of dataset/train_set)")
    print("   " + _format_metrics(holdout_metrics))
    print("2. Test-geometry set (dataset/new_geom)")
    print("   " + _format_metrics(test_metrics))
    print("=" * 72)
    print("Test-geometry breakdown")
    for name in sorted(test_metrics["by_source"]):
        print(f"   {name}: {_format_metrics(test_metrics['by_source'][name])}")
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {out_dir / 'model.pt'}")
    print(f"Training-fit {_format_metrics(fit_metrics)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        default=str(REPO_ROOT / "dataset" / "train_set"),
        help="Directory of training-geometry pklData files.",
    )
    parser.add_argument(
        "--test-dir",
        default=str(REPO_ROOT / "dataset" / "new_geom"),
        help="Directory of test-geometry validation pklData files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_DIR / "output"),
        help="Directory for the checkpoint, scaler, log, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hop", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--clip-value", type=float, default=30.0)
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ref-re", type=float, default=1e7)
    parser.add_argument("--ref-length", type=float, default=0.435762)
    parser.add_argument("--root-chord-ref", type=float, default=0.6536436)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on training-geometry cases, for a short smoke run.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
