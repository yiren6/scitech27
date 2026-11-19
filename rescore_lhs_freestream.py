"""Re-score saved HF/MF scaling checkpoints on lhs-free-stream geometries.

Original runs reported test metrics on cst-free-stream AoA-sweep
``*_validation.pkl`` files. This script leaves model weights untouched, updates
best/run ``report.json`` test sections, and rewrites aggregate scaling CSVs/JSONs
plus ``train_hf_scaling_report``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import pickle
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
MF_ROOT = Path("/home/yiren/Desktop/MF-VortexNet")
TRAIN_DIR = MF_ROOT / "dataset" / "train_set"
TEST_DIR = MF_ROOT / "dataset" / "new_geom" / "lhs-free-stream"
TEST_SET_LABEL = (
    "MF-VortexNet/dataset/new_geom/lhs-free-stream "
    "(*_vortexNetNewGeom.pkl; reported only; not used for selection)"
)

REF_RE = 10**7
REF_LENGTH = 0.435762
ROOT_CHORD_REF = 0.6536436

HF_SEEDS = (7, 27, 3407, 207, 353)
MF_SEEDS = (7, 27, 3407, 207, 353)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


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
                    setattr(sys.modules[".".join(parts[: index - 1])], parts[index - 1], mod)
        mod = sys.modules[module]
        if not hasattr(mod, name):
            setattr(mod, name, type(name, (_Stub,), {"__module__": module}))
        return getattr(mod, name)


def load_pickle_cases(directory: Path) -> list:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("pklData_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pklData_*.pkl in {directory}")
    cases = []
    for path in files:
        with path.open("rb") as handle:
            loaded = _NoSuaveUnpickler(handle).load()
        if not isinstance(loaded, list):
            raise TypeError(f"{path.name} did not contain a list")
        for case_index, case in enumerate(loaded):
            case.source_file = path.name
            case.case_index = case_index
        cases.extend(loaded)
        print(f"Loaded {len(loaded):4d} cases from {path.name}")
    for dataset_index, case in enumerate(cases):
        case.dataset_index = dataset_index
    print(f"Total cases from {directory}: {len(cases)}")
    return cases


def field_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
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


def evaluate_graphs(model, graphs, sources, device) -> dict:
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
    all_pred = np.concatenate(
        [arr for values in predictions_by_source.values() for arr in values]
    )
    all_true = np.concatenate(
        [arr for values in targets_by_source.values() for arr in values]
    )
    summary = field_metrics(all_pred, all_true)
    summary["n_cases"] = len(graphs)
    summary["by_source"] = {
        source: {
            **field_metrics(
                np.concatenate(predictions_by_source[source]),
                np.concatenate(targets_by_source[source]),
            ),
            "n_cases": len(predictions_by_source[source]),
        }
        for source in predictions_by_source
    }
    return summary


def case_metadata(case) -> dict:
    return {
        "dataset_index": int(case.dataset_index),
        "source_file": case.source_file,
        "case_index": int(case.case_index),
        "test_number": int(case.test_number),
        "mach": float(case.mach),
        "alpha": float(case.alpha),
        "Re": float(case.Re),
    }


# ---------------------------------------------------------------------------
# Multi-fidelity
# ---------------------------------------------------------------------------


def import_mf():
    # Prefer MF-VortexNet (exports DataParser); original cnfvortexnetff path is gone.
    sys.path.insert(0, str(MF_ROOT))
    sys.path.insert(0, str(MF_ROOT / "VortexNet"))
    from VortexNet import DataParser, GNN4  # noqa: WPS433

    return DataParser, GNN4


def build_mf_raw_graphs(parser, cases: list) -> list:
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
        print(f"  MF graphs {len(graphs)}/{len(cases)}")
    return graphs


def normalize_mf_graphs(parser_cls, raw_graphs: list, scaler_path: Path) -> list:
    scaler_data = parser_cls.load_scaler(str(scaler_path))
    # Dummy parser only needed for transform_single_graph API.
    parser = parser_cls(
        directory_path=str(TRAIN_DIR),
        ref_re=REF_RE,
        ref_length=REF_LENGTH,
        root_chord_ref=ROOT_CHORD_REF,
        apex_x=0.0,
        apex_y=0.0,
        apex_z=0.0,
        test_size=0.3,
        device="cpu",
        use_robust_scaler=bool(scaler_data.get("use_robust_scaler", False)),
        normalize_coordinates=bool(scaler_data.get("normalize_coordinates", True)),
    )
    out = []
    for graph in raw_graphs:
        out.append(parser.transform_single_graph(copy.deepcopy(graph), scaler_data))
    return out


def mf_model_from_report(GNN4, report: dict, device: torch.device):
    hp = report["hyperparameters"]
    return GNN4(
        node_in_channels=11,
        edge_in_channels=1,
        hidden_channels=int(hp["hidden_channels"]),
        out_channels=1,
        dropout_rate=float(hp["dropout"]),
        HEADS=int(hp["heads"]),
        ALPHA=float(hp["alpha"]),
        HOP=int(hp["hop"]),
    ).to(device)


def write_mf_aggregate(output_dir: Path, results: list[dict]) -> None:
    results = sorted(results, key=lambda item: (int(item["seed"]), float(item["fraction"])))
    _write_json(
        output_dir / "scaling_results.json",
        {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "rescored_test_set": TEST_SET_LABEL,
            "subset_sampling": "Nested random subsets of dataset/train_set per seed.",
            "validation_split": "70/30 training/validation within each nested subset.",
            "test_set": TEST_SET_LABEL,
            "test_geometry_used_for_selection": False,
            "results": results,
        },
    )
    csv_path = output_dir / "scaling_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
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


def rescore_mf(output_dir: Path, device: torch.device) -> None:
    DataParser, GNN4 = import_mf()
    print(f"\n=== Rescoring MF runs in {output_dir.name} ===")
    test_cases = load_pickle_cases(TEST_DIR)
    parser = DataParser(
        directory_path=str(TRAIN_DIR),
        ref_re=REF_RE,
        ref_length=REF_LENGTH,
        root_chord_ref=ROOT_CHORD_REF,
        apex_x=0.0,
        apex_y=0.0,
        apex_z=0.0,
        test_size=0.3,
        device=str(device),
        use_robust_scaler=False,
        normalize_coordinates=True,
    )
    print("Building raw MF test graphs")
    raw_test_graphs = build_mf_raw_graphs(parser, test_cases)
    test_sources = [case.source_file for case in test_cases]
    test_meta = [case_metadata(case) for case in test_cases]

    results = []
    run_dirs = sorted(output_dir.glob("seed_*/fraction_*/run"))
    for run_dir in run_dirs:
        report_path = run_dir / "report.json"
        model_path = run_dir / "model.pth"
        scaler_path = run_dir / "scaler.pkl"
        if not (report_path.is_file() and model_path.is_file() and scaler_path.is_file()):
            print(f"Skipping incomplete run {run_dir}")
            continue
        report = json.loads(report_path.read_text())
        fraction_dir = run_dir.parent
        print(
            f"MF {fraction_dir.parent.name}/{fraction_dir.name}: "
            f"old test RMSE {report.get('test_rmse')}"
        )
        graphs = normalize_mf_graphs(DataParser, raw_test_graphs, scaler_path)
        model = mf_model_from_report(GNN4, report, device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        test_metrics = evaluate_graphs(model, graphs, test_sources, device)
        report["n_test_cases"] = len(graphs)
        report["final_test_geometry"] = test_metrics
        report["test_rmse"] = float(test_metrics["rmse"])
        report["rescored_test_set"] = TEST_SET_LABEL
        report["rescored_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(report_path, report)

        split_path = fraction_dir / "split.json"
        if split_path.is_file():
            split = json.loads(split_path.read_text())
            split["test_case_count"] = len(test_cases)
            split["test_geometry_source_count"] = len({c.source_file for c in test_cases})
            split["test_geometry_cases"] = test_meta
            split["rescored_test_set"] = TEST_SET_LABEL
            _write_json(split_path, split)

        results.append(report)
        print(f"  -> new test RMSE {report['test_rmse']:.6g}")
        del model, graphs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError(f"No MF runs rescored under {output_dir}")
    write_mf_aggregate(output_dir, results)
    note = output_dir / "rescore_note.txt"
    note.write_text(
        f"Rescored {datetime.now().isoformat(timespec='seconds')}\n"
        f"Test set: {TEST_SET_LABEL}\n"
        f"Runs updated: {len(results)}\n"
        "Holdout/validation metrics and model weights unchanged.\n"
        "Previous AoA-sweep reports backed up under backup_wrong_aoa_test_20261118/.\n"
    )
    print(f"Wrote {output_dir / 'scaling_results.json'}")


# ---------------------------------------------------------------------------
# High-fidelity
# ---------------------------------------------------------------------------


def import_hf():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "VortexNet"))
    from train_hf_autoregressor import (  # noqa: WPS433
        build_hf_graphs,
        evaluate_split,
        make_model,
    )

    return build_hf_graphs, evaluate_split, make_model


def rescore_hf(device: torch.device, batch_size: int = 64) -> list[dict]:
    build_hf_graphs, evaluate_split, make_model = import_hf()
    print("\n=== Rescoring HF scaling best checkpoints ===")
    test_cases = load_pickle_cases(TEST_DIR)
    print("Building HF test graphs")
    test_graphs, test_sources = build_hf_graphs(test_cases, REF_RE)

    all_rows = []
    root = REPO / "train_hf_scaling_results"
    for seed in HF_SEEDS:
        seed_dir = root / f"train_hf_scaling_results_seed{seed}"
        if not seed_dir.is_dir():
            print(f"Missing {seed_dir}")
            continue
        seed_results = []
        for best_dir in sorted(seed_dir.glob("fraction_*/best")):
            report_path = best_dir / "report.json"
            model_path = best_dir / "model.pt"
            if not report_path.is_file() or not model_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            if "training_fraction" in report:
                fraction = float(report["training_fraction"])
            else:
                fraction = int(best_dir.parent.name.split("_")[1]) / 100.0
            print(
                f"HF seed {seed} {best_dir.parent.name}: "
                f"old test RMSE {report['test_report']['rmse']}"
            )
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            # Minimal namespace for make_model
            class _Args:
                pass

            args = _Args()
            kw = ckpt["model_kwargs"]
            args.hidden_channels = kw["hidden_channels"]
            args.dropout = kw["dropout_rate"]
            args.heads = kw["HEADS"]
            args.alpha = kw["ALPHA"]
            args.hop = kw["HOP"]
            model = make_model(args, device)
            model.load_state_dict(ckpt["model_state"])
            test_metrics = evaluate_split(
                model, test_graphs, test_sources, None, device, batch_size
            )
            # evaluate_split returns mse_normalized etc.; strip extras not in original if needed
            report["test_report"] = test_metrics
            report["test_geometry"] = test_metrics
            report["split"]["n_test_geometry"] = len(test_graphs)
            report["split"]["test_files"] = {
                source: sum(1 for s in test_sources if s == source)
                for source in sorted(set(test_sources))
            }
            report["rescored_test_set"] = TEST_SET_LABEL
            report["rescored_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(report_path, report)

            seed_results.append(
                {
                    "fraction": fraction,
                    "n_cases": int(report["split"]["n_fit"]) + int(report["split"]["n_holdout"]),
                    "n_fit": int(report["split"]["n_fit"]),
                    "n_holdout": int(report["split"]["n_holdout"]),
                    "best_trial": int(
                        report.get("hyperparameters", {}).get("best_trial")
                        or _best_trial_from_dirname(best_dir, report)
                    ),
                    "holdout_rmse": float(report["training_geometry_holdout"]["rmse"]),
                    "holdout_mae": float(report["training_geometry_holdout"]["mae"]),
                    "test_rmse": float(test_metrics["rmse"]),
                    "test_mae": float(test_metrics["mae"]),
                    "test_nrmse_std": float(test_metrics["nrmse_std"]),
                    "hyperparameters": json.dumps(
                        {
                            k: report["hyperparameters"][k]
                            for k in (
                                "alpha",
                                "dropout",
                                "heads",
                                "hidden_channels",
                                "hop",
                                "learning_rate",
                                "weight_decay",
                            )
                            if k in report["hyperparameters"]
                        },
                        sort_keys=True,
                    ),
                    "report": str(report_path),
                    "checkpoint": str(model_path),
                }
            )
            print(f"  -> new test RMSE {test_metrics['rmse']:.6g}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Recover best_trial numbers from existing scaling_results when possible
        old_summary_path = seed_dir / "scaling_results.json"
        old_by_frac = {}
        if old_summary_path.is_file():
            old = json.loads(old_summary_path.read_text())
            for item in old.get("results", []):
                old_by_frac[round(float(item["fraction"]) * 100)] = item
        for item in seed_results:
            key = round(float(item["fraction"]) * 100)
            if key in old_by_frac:
                item["best_trial"] = int(old_by_frac[key]["best_trial"])
                # keep holdout from old aggregate if report parsing was wrong
                item["holdout_rmse"] = float(old_by_frac[key]["holdout_rmse"])
                item["holdout_mae"] = float(old_by_frac[key]["holdout_mae"])

        seed_results.sort(key=lambda item: item["fraction"])
        _write_json(
            seed_dir / "scaling_results.json",
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "rescored_test_set": TEST_SET_LABEL,
                "subset_sampling": "Nested random subsets; seed-controlled.",
                "objective": "training_geometry_holdout.rmse",
                "test_set_used_for_selection": False,
                "results": seed_results,
            },
        )
        csv_path = seed_dir / "scaling_results.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_results[0].keys()))
            writer.writeheader()
            writer.writerows(seed_results)
        (seed_dir / "rescore_note.txt").write_text(
            f"Rescored {datetime.now().isoformat(timespec='seconds')}\n"
            f"Test set: {TEST_SET_LABEL}\n"
            "Only best/ checkpoints and aggregate scaling_results were updated.\n"
            "Per-trial report.json test metrics still reflect the old AoA-sweep set.\n"
        )
        for item in seed_results:
            all_rows.append({"seed": seed, **item})
        print(f"Wrote {seed_dir / 'scaling_results.json'}")
    return all_rows


def _best_trial_from_dirname(best_dir: Path, report: dict) -> int:
    # Fallback: study_summary
    summary = best_dir.parent / "study_summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text())
        best = data.get("best") or {}
        if "number" in best:
            return int(best["number"])
    return -1


def write_hf_average_report(rows: list[dict]) -> None:
    summary_dir = REPO / "train_hf_scaling_report"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = summary_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_fraction: dict[float, list[dict]] = {}
    for row in rows:
        by_fraction.setdefault(float(row["fraction"]), []).append(row)
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
            values = np.asarray([float(item[key]) for item in group], dtype=float)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for item in group:
            seed = int(item["seed"])
            record[f"seed{seed}_test_rmse"] = float(item["test_rmse"])
            record[f"seed{seed}_test_mae"] = float(item["test_mae"])
            record[f"seed{seed}_test_nrmse_std"] = float(item["test_nrmse_std"])
            record[f"seed{seed}_best_trial"] = int(item["best_trial"])
        summary_rows.append(record)

    csv_path = summary_dir / "seed_average.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (summary_dir / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")

    seeds = sorted({int(r["seed"]) for r in rows})
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 16,
        }
    )
    x = np.array([row["n_cases"] for row in summary_rows], dtype=float)
    tick_labels = [
        f"{int(round(row['fraction'] * 100))}%\n{row['n_cases']} cases"
        for row in summary_rows
    ]
    palette = {
        7: "#1f4e79",
        27: "#d35400",
        3407: "#2e8b57",
        207: "#8e44ad",
        353: "#16a085",
    }
    markers = {7: "o", 27: "s", 3407: "D", 207: "^", 353: "v"}
    panels = (
        ("test_rmse", "Test RMSE (ΔCp)", "Unseen-geometry RMSE"),
        ("test_mae", "Test MAE (ΔCp)", "Unseen-geometry MAE"),
        ("test_nrmse_std", "Test NRMSE (std)", "Unseen-geometry NRMSE"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 7.4))
    for ax, (key, ylabel, title) in zip(axes, panels):
        mean = np.array([row[f"{key}_mean"] for row in summary_rows])
        std = np.array([row[f"{key}_std"] for row in summary_rows])
        for seed in seeds:
            series = [row[f"seed{seed}_{key}"] for row in summary_rows]
            ax.plot(
                x,
                series,
                color=palette.get(seed, "#333333"),
                marker=markers.get(seed, "o"),
                ms=8,
                lw=1.5,
                label=f"Seed {seed}",
            )
        ax.plot(x, mean, color="#c0392b", lw=2.6, label=f"Mean of {len(seeds)} seeds", zorder=4)
        ax.fill_between(x, mean - std, mean + std, color="#c0392b", alpha=0.18, zorder=1)
        ax.set_xticks(x, tick_labels)
        ax.set_xlabel("Training snapshot")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    figure_path = fig_dir / "test_error_vs_training_snapshots.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    seed_header = " | ".join(f"Seed {s}" for s in seeds)
    lines = [
        f"# Training-snapshot scaling, averaged over {len(seeds)} seeds",
        "",
        "Each point is the Optuna winner for that snapshot (holdout RMSE). "
        "The plotted score is that winner's error on the **lhs-free-stream** "
        "generalized geometries "
        f"(`{TEST_DIR}`), rescored from saved checkpoints. "
        "Those cases were not used for training or trial selection.",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}. "
        "Previous AoA-sweep (`*_validation.pkl`) scores are in "
        "`backup_wrong_aoa_test_20261118/`.",
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
            f"| Snapshot | {seed_header} |",
            "|---:|" + "|".join(["---:"] * len(seeds)) + "|",
        ]
    )
    for row in summary_rows:
        pct = int(round(row["fraction"] * 100))
        cells = [
            f"{fmt(row[f'seed{s}_test_rmse'])} (trial {int(row[f'seed{s}_best_trial']):03d})"
            for s in seeds
        ]
        lines.append(f"| {pct}% | " + " | ".join(cells) + " |")
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
    (summary_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {summary_dir / 'report.md'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which",
        nargs="+",
        choices=("hf", "mf", "autohp", "all"),
        default=["all"],
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    which = set(args.which)
    if "all" in which:
        which = {"hf", "mf", "autohp"}
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Correct test dir: {TEST_DIR}")

    if "mf" in which:
        rescore_mf(REPO / "train_mf_scaling_results", device)
    if "autohp" in which:
        rescore_mf(REPO / "train_mf_scaling_autohp_results", device)
    if "hf" in which:
        rows = rescore_hf(device)
        write_hf_average_report(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
