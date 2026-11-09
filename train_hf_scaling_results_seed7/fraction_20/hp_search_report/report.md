# Hyperparameter search report: Seed 7, 20% training snapshot

Study `hf_scaling_20` from `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed7/fraction_20`. This study is the seed-7 nested snapshot that keeps 20% of `dataset/train_set`. Optuna minimized **holdout RMSE** on the random split of the training geometries. The sweep-60/70 wings in `dataset/new_geom` were scored after each trial and were **not** used for backpropagation or model selection.

## Search setup

- Trials launched: **100** (9 complete, 91 pruned)
- Random seed for the nested snapshot and the trial training: **7**
- Training snapshot: **20%** of the 600 training cases (120 cases; 84 fit / 36 holdout, about 70/30)
- Objective: `training_geometry_holdout.rmse`
- Checkpoint inside a trial: lowest training SmoothL1 on the fit split
- Fit / holdout / test-geometry cases: 84 / 36 / 44

Searched hyperparameters: `hidden_channels`, `heads`, `hop`, `alpha`, `dropout`, `learning_rate`, `weight_decay`.

## Outcome

Best trial: **022** (state `COMPLETE`, 171 epochs, 105981 parameters).

| Split | RMSE | NRMSE(std) | MAE |
|---|---:|---:|---:|
| Fit (70% of this snapshot) | 0.030531 | 0.064882 | 0.018348 |
| Holdout (selection metric) | 0.031876 | 0.065734 | 0.019549 |
| Test geometries (unseen wings) | 0.03591 | 0.071149 | 0.021738 |

Best-trial hyperparameters:

| Parameter | Value |
|---|---|
| hidden_channels | 22 |
| heads | 2 |
| hop | 13 |
| alpha | 0.606247 |
| dropout | 0.293774 |
| learning_rate | 0.009953 |
| weight_decay | 1.574e-04 |

## Optimization history

Each point is one Optuna trial. The red line is the best holdout RMSE seen so far. Open green markers are the test-geometry RMSE of completed trials; they are shown only as a diagnostic and do not choose the study winner. The vertical axis is logarithmic because a few pruned trials diverged.

![Holdout RMSE through the search](figures/01_optimization_history.png)

## Score distribution

The left panel is the Optuna objective. The right panel is the same trials scored on the unseen test geometries.

![Distribution of holdout and test RMSE](figures/02_objective_distribution.png)

## Hyperparameter effects

Each panel plots one searched parameter against holdout RMSE. Completed trials are darker; the best trial is ringed in red. Learning rate and weight decay are shown on a log scale.

![Hyperparameters versus holdout RMSE](figures/03_hyperparameter_effects.png)

## Top 5 completed trials

Ranking uses holdout RMSE only. Test-geometry RMSE is reported beside it so generalization can be compared without changing the ranking.

| Rank | Trial | Epochs | Params | Holdout RMSE | Test RMSE | Fit RMSE | hidden | heads | hop | α | dropout | lr | weight decay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 022 | 171 | 105981 | 0.031876 | 0.03591 | 0.030531 | 22 | 2 | 13 | 0.606 | 0.294 | 0.009953 | 1.574e-04 |
| 2 | 071 | 135 | 29059 | 0.033109 | 0.036673 | 0.031174 | 18 | 1 | 14 | 0.802 | 0.27 | 0.009409 | 5.088e-05 |
| 3 | 075 | 156 | 29059 | 0.036695 | 0.042175 | 0.034444 | 18 | 1 | 14 | 0.739 | 0.289 | 0.01565 | 1.482e-04 |
| 4 | 015 | 140 | 93147 | 0.037274 | 0.066944 | 0.037198 | 20 | 2 | 15 | 0.397 | 0.197 | 0.010642 | 4.242e-05 |
| 5 | 013 | 207 | 31705 | 0.03933 | 0.041673 | 0.038237 | 18 | 2 | 3 | 0.989 | 0.144 | 0.005253 | 1.740e-05 |

![Top trial RMSE comparison](figures/04_top5_metric_comparison.png)

The learning curves below are the per-epoch records stored with each trial. The kept checkpoint is still the lowest training SmoothL1, not the lowest point on the holdout curve.

![Top trial learning curves](figures/05_top5_training_curves.png)

## Pressure-field snapshots

The ten cases are the same reference cases used in the October 26 full-set and 55° NACA 0010 reports, so snapshots can be compared. The five training-geometry cases are sweep 55° NACA 0010; a given case is not necessarily inside this snapshot's holdout. The five test-geometry cases come from `dataset/new_geom` and were not used to train or to select the model. Each row is one case. Columns are predicted ΔCp and error (ground truth − prediction) for the top two completed trials. Predicted fields use that case's ground-truth ΔCp range. Error uses one symmetric scale for this figure.

### Reference training-geometry cases

| # | Geometry | Source pickle | Index in pickle | test_number | Mach | AoA [deg] | Re |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | sweep 55° NACA 0010 | `pklData_deltawing_sweep_55_naca_0010.pkl` | 24 | 25 | 0.48 | 6.38 | 6.670e+06 |
| 2 | sweep 55° NACA 0010 | `pklData_deltawing_sweep_55_naca_0010.pkl` | 27 | 28 | 0.41 | 6.86 | 6.550e+06 |
| 3 | sweep 55° NACA 0010 | `pklData_deltawing_sweep_55_naca_0010.pkl` | 30 | 31 | 0.42 | 12.45 | 9.860e+06 |
| 4 | sweep 55° NACA 0010 | `pklData_deltawing_sweep_55_naca_0010.pkl` | 35 | 36 | 0.49 | 13.26 | 8.160e+06 |
| 5 | sweep 55° NACA 0010 | `pklData_deltawing_sweep_55_naca_0010.pkl` | 13 | 14 | 0.4 | 16.71 | 9.370e+06 |

![Holdout pressure fields](figures/06_fields_holdout_top2.png)

### Test geometries

| # | Geometry | Source pickle | Index in pickle | test_number | Mach | AoA [deg] | Re |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | sweep 60° NACA 0013 | `pklData_deltawing_sweep_60_naca_0013_validation.pkl` | 1 | 2 | 0.4 | 7 | 7.000e+06 |
| 2 | sweep 60° NACA 3416 | `pklData_deltawing_sweep_60_naca_3416_validation.pkl` | 4 | 5 | 0.4 | 13 | 7.000e+06 |
| 3 | sweep 70° NACA 0013 | `pklData_deltawing_sweep_70_naca_0013_validation.pkl` | 6 | 7 | 0.4 | 17 | 7.000e+06 |
| 4 | sweep 70° NACA 3416 | `pklData_deltawing_sweep_70_naca_3416_validation.pkl` | 8 | 9 | 0.4 | 21 | 7.000e+06 |
| 5 | sweep 60° NACA 0013 | `pklData_deltawing_sweep_60_naca_0013_validation.pkl` | 10 | 11 | 0.4 | 25 | 7.000e+06 |

![Test-geometry pressure fields](figures/07_fields_testgeom_top2.png)

## Best trial on unseen geometries

Per-file RMSE on `dataset/new_geom`. These wings were not in the training split and were not used to pick the trial.

| Geometry file | RMSE | NRMSE(std) | MAE | Cases |
|---|---:|---:|---:|---:|
| `pklData_deltawing_sweep_60_naca_0013_validation.pkl` | 0.028997 | 0.064142 | 0.016907 | 11 |
| `pklData_deltawing_sweep_60_naca_3416_validation.pkl` | 0.036483 | 0.068953 | 0.021961 | 11 |
| `pklData_deltawing_sweep_70_naca_0013_validation.pkl` | 0.034103 | 0.070729 | 0.022751 | 11 |
| `pklData_deltawing_sweep_70_naca_3416_validation.pkl` | 0.042698 | 0.083387 | 0.025332 | 11 |

## Files

- Trial table: `trials.csv`
- Study summary: `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed7/fraction_20/study_summary.json`
- Best checkpoint: `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed7/fraction_20/best/model.pt`
- Field-case identities: `field_cases.json`
