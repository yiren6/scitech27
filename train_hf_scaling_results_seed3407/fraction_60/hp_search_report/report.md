# Hyperparameter search report: Seed 3407, 60% training snapshot

Study `hf_scaling_60` from `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed3407/fraction_60`. This study is the seed-3407 nested snapshot that keeps 60% of `dataset/train_set`. Optuna minimized **holdout RMSE** on the random split of the training geometries. The sweep-60/70 wings in `dataset/new_geom` were scored after each trial and were **not** used for backpropagation or model selection.

## Search setup

- Trials launched: **100** (16 complete, 84 pruned)
- Random seed for the nested snapshot and the trial training: **3407**
- Training snapshot: **60%** of the 600 training cases (360 cases; 252 fit / 108 holdout, about 70/30)
- Objective: `training_geometry_holdout.rmse`
- Checkpoint inside a trial: lowest training SmoothL1 on the fit split
- Fit / holdout / test-geometry cases: 252 / 108 / 44

Searched hyperparameters: `hidden_channels`, `heads`, `hop`, `alpha`, `dropout`, `learning_rate`, `weight_decay`.

## Outcome

Best trial: **059** (state `COMPLETE`, 137 epochs, 22579 parameters).

| Split | RMSE | NRMSE(std) | MAE |
|---|---:|---:|---:|
| Fit (70% of this snapshot) | 0.023019 | 0.04675 | 0.015116 |
| Holdout (selection metric) | 0.022561 | 0.047681 | 0.015099 |
| Test geometries (unseen wings) | 0.025874 | 0.051264 | 0.0169 |

Best-trial hyperparameters:

| Parameter | Value |
|---|---|
| hidden_channels | 18 |
| heads | 1 |
| hop | 8 |
| alpha | 0.840241 |
| dropout | 0.076271 |
| learning_rate | 0.016365 |
| weight_decay | 1.148e-04 |

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
| 1 | 059 | 137 | 22579 | 0.022561 | 0.025874 | 0.023019 | 18 | 1 | 8 | 0.84 | 0.076 | 0.016365 | 1.148e-04 |
| 2 | 008 | 129 | 45720 | 0.023036 | 0.026611 | 0.023648 | 17 | 2 | 9 | 0.793 | 0.068 | 0.019077 | 1.266e-06 |
| 3 | 022 | 133 | 20419 | 0.026681 | 0.031299 | 0.028177 | 18 | 1 | 6 | 0.938 | 0.267 | 0.044693 | 1.073e-06 |
| 4 | 021 | 93 | 20419 | 0.029206 | 0.034718 | 0.030772 | 18 | 1 | 6 | 0.931 | 0.262 | 0.047941 | 1.038e-06 |
| 5 | 027 | 121 | 100075 | 0.029755 | 0.055265 | 0.0298 | 31 | 1 | 11 | 0.612 | 0.093 | 0.012703 | 1.131e-04 |

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
| `pklData_deltawing_sweep_60_naca_0013_validation.pkl` | 0.020914 | 0.046262 | 0.01395 | 11 |
| `pklData_deltawing_sweep_60_naca_3416_validation.pkl` | 0.0252 | 0.047628 | 0.016804 | 11 |
| `pklData_deltawing_sweep_70_naca_0013_validation.pkl` | 0.025811 | 0.053532 | 0.018178 | 11 |
| `pklData_deltawing_sweep_70_naca_3416_validation.pkl` | 0.030645 | 0.059848 | 0.018668 | 11 |

## Files

- Trial table: `trials.csv`
- Study summary: `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed3407/fraction_60/study_summary.json`
- Best checkpoint: `/Users/yirenshen/Desktop/scitech27/train_hf_scaling_results_seed3407/fraction_60/best/model.pt`
- Field-case identities: `field_cases.json`
