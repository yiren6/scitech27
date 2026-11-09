# scitech27 results

## Where to retrieve the split

The split is the list of which CFD cases were used to fit the model and which were held out. It is not inside `hp_search_report/` or `train_hf_scaling_report/`. Those folders only store scores. The unseen sweep-60° and sweep-70° wings in `dataset/new_geom` are never listed in these split files; they are the test geometries scored after training.

### Scaling runs (seeds 7, 27, 3407)

Retrieve the split from the CSV in that seed’s folder:

- `train_hf_scaling_results_seed7/train_set_split.csv`
- `train_hf_scaling_results_seed27/train_set_split.csv`
- `train_hf_scaling_results_seed3407/train_set_split.csv`

One file holds all four snapshots for that seed. Filter on `fraction`:

| `fraction` | Snapshot | Rows | Fit (`split=training`) | Holdout (`split=test`) |
|---:|---:|---:|---:|---:|
| 0.2 | 20% | 120 | 84 | 36 |
| 0.4 | 40% | 240 | 168 | 72 |
| 0.6 | 60% | 360 | 252 | 108 |
| 0.8 | 80% | 480 | 336 | 144 |

`split=training` is the fit set. `split=test` is the holdout drawn from the same snapshot (`test_size` 0.3). Optuna selects the trial with that holdout. The three CSVs are different: each seed draws its own nested subsets of the 600 training cases.

Columns: `fraction`, `split`, `dataset_index`, `graph_index`, `source_file`, `case_index`, `test_number`, `mach`, `alpha`, `Re`. `source_file` plus `case_index` identifies the case inside the training pickle. `graph_index` is the position of that case when the graphs were built.

### October 26 Optuna runs

Each study folder has the same split in two forms: `train_set_split.txt` (table) and `train_set_split.json` (one record per case). `training` is the fit split. `test` is the holdout of those same geometries.

- `Oct26_Optuna_VortexNet_all_geoms/train_set_split.txt` — all 15 training geometries, seed 3407, `test_size` 0.3, 420 fit / 180 holdout.
- `Oct26_Optuna_Vortexnet_SF_550010_7030split/train_set_split.txt` — sweep 55° NACA 0010 only, seed 3407, `test_size` 0.3, 28 fit / 12 holdout.
- `Oct26_Optuna_Vortexnet_SF_550010_8020split/train_set_split.txt` — sweep 55° NACA 0010 only, seed 3407, `test_size` 0.3, 28 fit / 12 holdout.

### CNF VortexNet

`vortexnet_cnf_train_set_split.txt` at this repo root. Seed 3407, `test_size` 0.3, 420 fit / 180 holdout of the same geometries. Membership is stored as `graph_index` lists per source pickle. K-fold validation stays inside the fit split.

## What each folder is

- `Oct26_Optuna_VortexNet_all_geoms` — Oct 26 single-fidelity VortexNet (CFD in, ΔCp out) on all training geometries. Hyperparameter report: `hp_search_report/`.
- `Oct26_Optuna_Vortexnet_SF_550010_7030split` — same model, sweep 55° NACA 0010 only.
- `Oct26_Optuna_Vortexnet_SF_550010_8020split` — same model, sweep 55° NACA 0010 only.
- `train_hf_scaling_results_seed7`, `train_hf_scaling_results_seed27`, `train_hf_scaling_results_seed3407` — Optuna search at each nested snapshot. Per-snapshot report: `fraction_20/hp_search_report/` (and `fraction_40`, `fraction_60`, `fraction_80`).
- `train_hf_scaling_report` — mean test error of the three seeds versus snapshot size. Figure: `figures/test_error_vs_training_snapshots.png`.
