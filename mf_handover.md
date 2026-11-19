# Multi-fidelity scaling runner handover

## Purpose and location

`train_mf_scaling.py` runs a multi-fidelity training-data scaling experiment in
`scitech27`, using the existing model, parser, and training implementation from
`/home/yiren/cnfvortexnetff`. It does not modify files in `cnfvortexnetff`.

The experiment varies the fraction of training geometries used for fitting,
repeats each fraction with five data-split seeds, selects hyperparameters using
a holdout from the training geometries, and reports final performance on
separate new geometries.

## Data and defaults

- Training data: `dataset/train_set` (currently 600 cases).
- Final test data: `dataset/new_geom` validation pickles (currently 44 cases).
- Seeds: `7, 27, 3407, 207, 353`.
- Fractions: `0.06, 0.10, 0.16, 0.20, 0.40, 0.60, 0.80, 1.0`.
- Default output directory: `train_mf_scaling_results`.
- Defaults: 10 Optuna trials per seed/fraction, 3000 epochs, four CV folds,
  30% training-geometry holdout.

At each fraction, the subset is selected by taking the first N entries of a
seeded permutation of the full training set. Since the same permutation is
reused for each fraction of a seed, the subsets are nested. The selected cases
are split into a fit partition and a training-geometry holdout using
`random_state = seed + round(fraction * 100)`. The default 30% holdout is the
metric Optuna minimizes; the separate 4-fold CV remains part of the original
CNF trainer protocol on the fit partition.

## Training and selection design

The runner imports `DataParser`, `GNN4`, and `train_model_k_fold` from
`cnfvortexnetff`. Pickles are loaded with a custom unpickler that stubs
unavailable SUAVE classes not needed to construct the panel graphs.

For each fraction:

1. Build graphs for the fit cases, holdout cases, and new-geometry cases.
2. Fit normalization using only the fit partition. Normalize both evaluation
   groups with that scaler.
3. For each Optuna trial, call the original `train_model_k_fold` on the fit
   graphs with four folds. Evaluate the returned model on the training-
   geometry holdout.
4. Return holdout RMSE to Optuna. Trial attributes record fold validation
   losses, mean CV validation loss, and holdout RMSE/MSE.
5. After selecting the completed trial with the lowest holdout RMSE, load its
   checkpoint and evaluate it on `dataset/new_geom`. Final test metrics do not
   influence trial selection and are not recorded as trial attributes.

The new-geometry graphs are built and normalized before optimization so they
are ready for the final scoring step, but the Optuna objective does not evaluate
them. The final report also records `test_geometry_used_for_selection: false`.

Optuna searches learning rate in `[1e-4, 8e-3]` and decay in `[1e-6, 1e-5]`.
Other model/training settings remain fixed to the runner constants, including
GNN4 with 11 node input channels, 1 edge input channel, 63 hidden channels,
four heads, dropout `0.08504544001617964`, and the existing physical-loss,
noise, clipping, and reference-geometry settings in `train_mf_scaling.py`.

## Split records and outputs

For each seed/fraction, under
`train_mf_scaling_results/seed_<seed>/fraction_<percent>/`:

- `split.json`: selected, fit, holdout, and new-geometry case metadata.
- `train_validation_split.csv`: outer fit/holdout assignments.
- `cv_fold_split.csv`: fit-case assignments for each inner CV fold.
- `study.db` and `study_summary.json`: Optuna study and trial details.
- `best/model.pth`, `best/scaler.pkl`, `best/report.json`: selected model,
  its scaler, and metrics.
- `trials/`: temporary per-trial checkpoints, removed after a fraction
  completes.

The output root also contains `train_log.txt`, aggregate
`scaling_results.json` and `scaling_results.csv`, and a cumulative
`train_validation_split.csv`. The aggregate JSON marks that test geometries
were not used for selection. Nested metric objects in the CSV are JSON strings.

The study is persisted in SQLite and loaded on rerun; the runner starts only
the number of additional trials needed to reach `--n-trials`. Existing trial
checkpoints are cleaned up after successful scoring, while the selected model
and report remain in `best/`.

## Running

```bash
source /home/yiren/miniconda3/bin/activate vortexnet
cd /home/yiren/scitech27
python train_mf_scaling.py
```

For example, to run only seeds 207 and 353, keep the same default fractions and
other settings:

```bash
python train_mf_scaling.py --seeds 207 353
```

Other supported options include `--train-dir`, `--test-dir`, `--output-dir`,
`--fractions`, `--n-trials`, `--epochs`, `--folds`, `--test-size`, and
`--device`. Run `python train_mf_scaling.py --help` for the CLI details.

## Validation status

Syntax and CLI checks passed. Smoke checks successfully loaded and constructed
one graph from each dataset, verified outer and CV split-manifest writing,
checked metric aggregation, and confirmed the Optuna objective/test-scoring
ordering. The full scaling training sweep has **not** been run yet.

---

## Revision log — 2026-11-18

Implemented on **2026-11-18**. The original runs below had scored the wrong
generalized-geometry test set. Model weights were kept; test metrics, reports,
and figures were regenerated on the correct lhs-free-stream cases.

### Wrong vs correct test data

- **Incorrect (original scoring):** `*_validation.pkl` AoA-sweep freestreams
  (fixed \(M=0.4\), \(\mathrm{Re}=7\times10^6\)), historically under
  `dataset/new_geom` / `cst-free-stream(aoa-sweep)`.
- **Correct:**  
  `/home/yiren/Desktop/MF-VortexNet/dataset/new_geom/lhs-free-stream`  
  (`*_vortexNetNewGeom.pkl`, 4 files × 11 cases = 44 LHS freestreams).
- **Training geometries (unchanged):**  
  `/home/yiren/Desktop/MF-VortexNet/dataset/train_set` (15 pickles, 600 cases).

### Backup of prior accuracy / history

Directory: `backup_wrong_aoa_test_20261118/`

Contains prior `report.json`, `scaling_results.*`, logs, splits, and
`train_hf_scaling_report/` from before rescoring. Model weights were **not**
copied; they remained in the live result trees.

Also: `plots/independent_variable_pairplot_aoa_sweep_backup.png` is the
previous pair plot that used the AoA-sweep validation set.

### Rescoring from saved checkpoints

Script: `rescore_lhs_freestream.py`

Re-evaluated saved weights on lhs-free-stream without retraining:

| Tree | What was updated |
|---|---|
| `train_mf_scaling_results/` | 40 `run/report.json` + aggregates |
| `train_mf_scaling_autohp_results/` | 40 `run/report.json` + aggregates |
| `train_hf_scaling_results/` | 35 `best/report.json` + per-seed aggregates |

Holdout / validation metrics and checkpoints were left unchanged. HF
**per-trial** `report.json` test fields still reflect the old AoA-sweep set;
only best-checkpoint scores feed the aggregates.

```bash
python rescore_lhs_freestream.py            # hf + mf + autohp
python rescore_lhs_freestream.py --which hf
```

### Default paths and loaders (future runs)

Updated so new runs do not re-pick the AoA-sweep validation files:

- `train_mf_scaling.py`, `train_mf_scaling_autohp.py`, `train_hf_scaling.py`,
  `train_hf_autoregressor.py`
- Defaults: absolute `MF-VortexNet` `train_set` and `lhs-free-stream`
- Loaders accept `*_vortexNetNewGeom.pkl` (not only `*validation*`)
- MF import root set to `/home/yiren/Desktop/MF-VortexNet` (old
  `/home/yiren/cnfvortexnetff` path is gone on this machine)

### Reports and scaling figures

Script: `report_all_scaling.py`

Writes durable cross-seed reports:

- `train_hf_scaling_report/`
- `train_mf_scaling_report/`
- `train_mf_scaling_autohp_report/`

Each report folder has `report.md`, `seed_average.csv`, `per_seed.json`, and
**one figure per metric** (not a 1×3 panel):

- `figures/test_rmse_vs_training_snapshots.png`
- `figures/test_mae_vs_training_snapshots.png`
- `figures/test_nrmse_std_vs_training_snapshots.png`

Figure style (2026-11-18):

- Labels **Run 1 … Run 5** (sorted seed order: 7, 27, 207, 353, 3407), not
  “Seed \(x\)”
- Log–log axes on both \(x\) and \(y\)
- Individual runs at **50%** opacity (`alpha=0.5`)
- Mean of 5 runs solid; mean linewidth = **1.5×** individual-run linewidth
- ±1 sample-std band around the mean retained

HF/MF mean overlays (also one file per metric):

- `figures/hf_mf_test_rmse_overlay.png`
- `figures/hf_mf_test_mae_overlay.png`
- `figures/hf_mf_test_nrmse_std_overlay.png`

```bash
python report_all_scaling.py
```

### Independent-variable pair plot

Script: `plots/plot_independent_variable_pairplot.py`

- Defaults point at `train_set` + `lhs-free-stream`
- Loader accepts `*_vortexNetNewGeom.pkl`
- Legend: **Training** / **Test**
- Output: `plots/independent_variable_pairplot.png` (600 + 44 cases)

```bash
python plots/plot_independent_variable_pairplot.py
```

### Docs touched

- `readme.md` — report folder paths and lhs-free-stream test-set wording
- `mf_scaling_analysis.ipynb`, `mf_scaling_analysis_with_hpo.ipynb` —
  markdown updated to mention lhs-free-stream
- This file (`mf_handover.md`) — this revision log

### Note on earlier sections of this handover

Sections above this revision log still describe the original MF runner design
and may mention `dataset/new_geom` validation pickles / `cnfvortexnetff`. For
**current** data paths, scoring, and reporting, prefer this 2026-11-18 section.
