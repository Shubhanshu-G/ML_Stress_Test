# What drives generalization? A fixed Random Forest on UCI Superconductivity

**Question.** With the model held fixed, which properties of the data decide how well it generalizes?

## Setup

- **Data:** UCI Superconductivty Data. 21,263 materials, 81 features derived from the chemical formula, target is critical temperature Tc (K). File hashes and download date are in `config.yaml`.
- **Model:** `RandomForestRegressor`, 200 trees, no tuning, identical for every run.
- **Metric:** RMSE in K. MAE and R² are in `results/raw.csv`. R² is not used for the OOD split because its test range is narrow.
- **Test sets:** `grouped` (primary: all rows of one formula go to the same side, 20% test), `ood` (train on Tc <= 74 K, test on Tc > 74 K), `random_iid` (baseline only).
- **Sweeps:** training size (2 to 100%), label noise, input noise, masked features, and a size x label-noise grid. 10 seeds per condition, reported as mean +/- std over seeds.
- **Pre-commitment:** levels, splits, seeds and metric were committed in `config.yaml` (tag `config-frozen`) before any comparison ran. Items marked *post-hoc* were added afterwards. The code that produced `results/raw.csv` is commit `979d3f8`. Final package: commit `<final hash>`.

## Results (test RMSE in K, mean +/- std over 10 seeds)

| Condition | Grouped | OOD |
| --- | --- | --- |
| Baseline (full data, no perturbation) | 10.42 +/- 0.05 | 39.17 +/- 0.09 |
| Training size 50% | 11.11 +/- 0.31 | 39.90 +/- 0.39 |
| Training size 10% | 13.43 +/- 0.32 | 42.17 +/- 1.37 |
| Training size 2% | 16.77 +/- 0.60 | 46.22 +/- 1.19 |
| Label noise 0.4 | 12.11 +/- 0.34 | 39.15 +/- 0.23 |
| Input noise 0.4 | 15.61 +/- 0.12 | 39.75 +/- 0.36 |
| 10% features masked | 11.72 +/- 0.14 | 41.61 +/- 0.09 |
| 50% features masked | 15.71 +/- 0.17 | 46.75 +/- 0.17 |

Random-split baseline: 9.43 +/- 0.01. Always predicting the training mean gives 34.1 (grouped) and 72.4 (OOD). All conditions: `results/summary.csv`. Curves: `figures/`.

## What drives generalization

1. **Whether the test materials resemble the training set matters more than anything perturbed.** Same model: 9.4 K (random split), 10.4 K (grouped), 39.2 K (OOD). About 90% of the OOD squared error is systematic under-prediction of hot materials (seed 0: bias -28 K for Tc 74 to 100 K, -64 K above 100 K), not scatter.
2. **On the grouped split, less data, missing features and input noise cost about 5 to 6 K each at the levels tried.** Label noise costs 1.7 K at the highest level (0.4) and at most 0.13 K up to 0.1, comparable to the seed spread. The levels are not commensurable (masking is applied to train and test, noise to train only), so this ranks the tested levels, not the factors in general. The learning curve has not saturated: 16.8 K at 2% of the data, 10.4 K at 100%.
3. **On OOD, label and input noise barely matter (at most +0.7 K); missing features (+7.6 K at 50%) and tiny training sets (+7.1 K at 2%) do.** Doubling the data from 50% to 100% gains only 0.7 K.
4. *(post-hoc)* **The data sets a floor on error.** 37.5% of rows belong to formulas that appear more than once, with identical features but different Tc (pooled within-formula std 6.85 K). `Y1Ba2Cu3O7` has 110 rows spanning 30 to 130 K. A model that sees only formula-derived features must give all of them one prediction.
5. *(post-hoc)* **Chemistry family matters, and it is confounded with Tc.** On the grouped split, Cu+O compounds have 13.6 K RMSE against 5.6 K for the rest, and cuprates with Tc <= 74 K are already at 12.7 K. 841 of the 842 test rows above 74 K are cuprates, so "hot" and "cuprate" cannot be separated here. Hot cuprates cost 14.9 K when other hot cuprates are in the training set (grouped) and 39.0 K when none are (OOD), on comparable but not identical test rows. The OOD failure is mostly missing exposure to the hot range.
6. *(post-hoc)* **The choice of test partition moves the headline more than the model's randomness does.** Ten other grouped partitions give 9.00 to 10.47 K (mean 9.68, std 0.42), while the std across seeds on one partition is 0.05. The committed partition is the highest of the ten. One formula (`Tl2Ba2Cu1O6`, 45 rows) accounts for 16.7% of its squared error, and without it the RMSE is 9.61 K. Comparisons between conditions are paired (same test set), but absolute levels depend on the partition. The leakage effect (random vs grouped) is small and not resolved by this evidence: 0.25 K against the grouped mean, within the partition spread.

## Failure cases (`failure_cases/`, seed 0)

- **Hot materials are under-predicted.** Grouped bias is -9.6 K for Tc > 100 K and +1.9 K for Tc <= 10 K. On OOD, materials at 131 to 143 K are predicted at about 44 to 52 K. Every perturbation pushes hot materials lower: bias for Tc > 100 K is -33 K at 2% data and -29 K at 50% masking.
- **Repeated formulas with a wide Tc spread get one prediction.** `Tl2Ba2Cu1O6` is predicted at 28.5 K for 45 rows with Tc between 10 and 117 K.
- **Single-element `C1`** (22 rows, Tc 0.4 to 32 K) is predicted at about 38 K on average, 5.3% of the grouped squared error.
- **Feature shift.** The worst 5% of predictions differ from the rest by about 0.6 to 1.6 standard deviations in a few features (thermal conductivity spread, valence, density range). The worst rows are mostly hot cuprates, so this describes what they look like, not a cause.

## Limits

- One dataset and one model. A random forest averages training labels, so it cannot predict above the highest training Tc; the size of the OOD failure is partly a property of the model class. Other models were not tested, by design.
- The train/test partition is fixed. Error bars cover model, noise and subsampling randomness, not the partition (see item 6).
- Perturbation levels are not commensurable. Input noise is applied to training data only, masking to train and test, and label noise is not clipped.
- Items 4 to 6 and the failure-case analysis are post-hoc and, where noted, from single runs (seed 0).
