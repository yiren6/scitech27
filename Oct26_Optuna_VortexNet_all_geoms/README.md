# Oct 26 Optuna search: VortexNet on all training geometries

Results of study `hf_autoregressor` (mode `full`), finished 2026-10-26. Optuna minimized **holdout RMSE** of a GNN4 autoregressor that predicts high-fidelity CFD ΔCp on the delta-wing panel graph.

- Training geometries come from `dataset/train_set` (600 cases, 15 wings at 55°, 65°, and 75° sweep). A random split with seed 3407 and `test_size` 0.3 gives **420 fit** cases and **180 holdout** cases. The holdout is the Optuna objective. The checkpoint kept inside a trial is the lowest training SmoothL1 on the fit split.
- Unseen wings in `dataset/new_geom` (44 cases: sweep 60° and 70°, NACA 0013 and 3416) are scored after every trial. They are not used for training or for choosing the winner.
- 100 trials were launched. 13 finished (`COMPLETE`); 87 were pruned. Best trial is **036** (holdout RMSE 0.013831, test-geometry RMSE 0.015805).

This directory stores reports and the split. Model checkpoints (`model.pt`) are not included. Paths inside the JSON still point at `/home/yiren/scitech27/optuna_results/`.

## Top-level files

| File | What it contains |
|---|---|
| `study_summary.json` | Whole study in one file: name, minimize direction, objective (`training_geometry_holdout.rmse`), mode, trial count, the winning trial, and one record per trial (state, objective value, searched hyperparameters, and holdout / test RMSE). Updated 2026-10-26T20:47:47. Use this to rank trials without opening each folder. |
| `search_log.txt` | Console log of the run. Header records device, epoch budget (3000), batch size (64), and the loaded pickle counts. Then one block per trial: hyperparameters, parameter count, a line per epoch (`train_smoothl1`, learning rate, holdout RMSE / NRMSE / MAE / MSE), and a save line for the completed or pruned trial plus its test-geometry RMSE. The last block names trial 36 as the winner. |
| `train_set_split.json` | The random split of `dataset/train_set`. `training` is the 420-case fit list; `test` is the 180-case holdout of the same geometries. `dataset/new_geom` is not listed. Each case has `dataset_index`, `graph_index`, `source_file`, `case_index`, `test_number`, Mach, angle of attack, and Reynolds number. Seed 3407, `test_size` 0.3. |
| `train_set_split.txt` | The same split as a tab-separated table, with a short header. |

## `trial_000` … `trial_099`

One directory per Optuna trial, numbered from 0. Each directory contains only `report.json` (the checkpoint is not here).

`report.json` is the full record for that trial:

- `description` and `error_definition` — the model predicts raw HF ΔCp; RMSE, NRMSE(std), NRMSE(range), MAE, and MSE are defined there.
- `status` — `complete` or `pruned`.
- `checkpoint_selected_by` — lowest training SmoothL1 on the fit split.
- `hyperparameters` — searched values (`hidden_channels`, `heads`, `hop`, `alpha`, `dropout`, `learning_rate`, `weight_decay`) plus fixed settings (`noise`, `clip_value`, `batch_size`, `epochs_requested`, `patience`, `scheduler_patience`, `penalty_weight`, `seed`, `test_size`) and `n_parameters`.
- `split` — case counts (420 / 180 / 44) and how many cases of each source pickle fell into fit, holdout, and test geometries.
- `training_report` — aggregate ΔCp errors on the fit split and the holdout, broken down by source pickle, plus the per-epoch `history` (epoch, training SmoothL1, learning rate, holdout RMSE / NRMSE / MSE).
- `training_fit` and `training_geometry_holdout` — the same final fit and holdout aggregates, repeated at the top level.
- `test_report` and `test_geometry` — the same aggregate errors on the 44 unseen-geometry cases, overall and per validation pickle.

## `best/`

`best/report.json` is the report of the winning trial (trial 036). Same schema as `trial_036/report.json`. The matching `best/model.pt` checkpoint is not in this copy.

## `hp_search_report/`

Written summary of the search. Figures cited in the markdown (`figures/01_…` through `figures/07_…`) are not stored in this directory.

| File | What it contains |
|---|---|
| `report.md` | Narrative report: search setup, best-trial metrics and hyperparameters, top 5 completed trials, the ten pressure-field cases used in the plots, and per-file RMSE of the best trial on the four unseen geometries. |
| `trials.csv` | One row per trial. Columns: trial number, Optuna state, status, epochs run, parameter count, holdout / fit / test RMSE (plus NRMSE and MAE), and the seven searched hyperparameters. |
| `field_cases.json` | The ten cases plotted in the report: five holdout cases from sweep 55° NACA 0010, and five test-geometry cases from the 60° and 70° validation pickles. Each entry has source pickle, case index, `test_number`, Mach, angle of attack, and Reynolds number. |
