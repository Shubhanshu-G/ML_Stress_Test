# What drives generalization? A fixed Random Forest on UCI Superconductivity

**Question.** With the model held fixed, which properties of the data are associated with how well it generalizes?

## Setup

- **Data:** UCI Superconductivty Data (id 464). 21,263 materials, 81 formula-derived features, target is critical temperature Tc (K). File hashes and download date are in `config.yaml`.
- **Model and metric:** `RandomForestRegressor`, 200 trees, no tuning, identical in every run. RMSE in K is primary. MAE and R² are in `results/raw.csv`; R² is not used for OOD because its test range is narrow.
- **Test sets:** `grouped` (primary: all rows of a formula on one side, 20% test), `ood` (train on Tc <= 74.0 K, test above; 74.0 K is the 80th percentile of Tc over the full dataset), `random_iid` (baseline only).
- **Perturbations:** training size (2 to 100%); label noise (Gaussian, std = level x std of training Tc, training labels only); input noise (Gaussian, std = level x each feature's training std, training features only); masked features (random cells, train and test, filled with training means). One factor at a time plus a 3 x 3 size x label-noise grid. 43 conditions x 10 seeds = 430 runs, reported as mean +/- std over seeds.
- **Pre-commitment:** the design was committed in `config.yaml` (`a085154`, 2026-09-30) and has not changed. The code that produced `results/raw.csv` is `979d3f8`. Final package: tag `final`. No hypotheses or pass/fail thresholds were written before the run. Items marked *post-hoc* were found after seeing results. Timeline and disclosures (baseline check, smoke runs, one stopped launch): `PROTOCOL.md`.

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

## What the results show

1. **Test-set composition is associated with the largest differences.** Same model: 9.4 K (random), 10.4 K (grouped), 39.2 K (OOD). About 90% of the OOD squared error is systematic under-prediction of hot materials (seed 0: bias -28 K for Tc 74 to 100 K, -64 K above 100 K), not scatter.
2. **Grouped split:** 2% of the data, 50% masked features and input noise 0.4 each raised RMSE by 5.2 to 6.4 K at the most-damaged level tried. Label noise added 1.7 K at 0.4 and at most 0.13 K up to 0.1, comparable to the seed spread. Levels are not commensurable, so this ranks the tested levels, not the factors in general. The learning curve has not saturated (16.8 K at 2%, 10.4 K at 100%).
3. **OOD split:** label and input noise changed RMSE by at most 0.7 K. Masking (+7.6 K at 50%) and tiny training sets (+7.1 K at 2%) did more. Going from 50% to 100% of the data gained only 0.7 K.
4. *(post-hoc)* **Repeated formulas set an error floor.** 37.5% of rows belong to formulas that appear more than once, with identical features but different Tc (pooled within-formula std 6.85 K). `Y1Ba2Cu3O7` has 110 rows spanning 30 to 130 K, and a model on formula-derived features must give them one prediction.
5. *(post-hoc)* **Chemistry is confounded with Tc.** Cu+O compounds have 13.6 K RMSE against 5.6 K for the rest (12.7 K for Tc <= 74). 841 of the 842 grouped test rows above 74 K are cuprates, so "hot" and "cuprate" cannot be separated here. Hot cuprates score 14.9 K when other hot cuprates are in training (grouped) and 39.0 K when none are (OOD), on comparable but not identical rows. This suggests, not shows, that the OOD failure reflects missing exposure to the hot range.
6. *(post-hoc)* **The test partition moved the headline more than model randomness.** Across ten grouped partitions, baseline RMSE ranges from 9.00 to 10.47 K (mean 9.68, std 0.42; the committed partition is the highest), against a std of 0.05 across seeds on one partition. The committed random-vs-grouped gap (0.98 K) probably overstates the typical one; the random split's own partition spread was not measured. One formula (`Tl2Ba2Cu1O6`, 45 rows) accounts for 16.7% of the committed squared error (RMSE 9.61 K without it).

## Failure cases (`failure_cases/`, seed 0)

- **Hot materials are under-predicted.** Grouped bias is -9.6 K for Tc > 100 K. On OOD, materials at 131 to 143 K are predicted at about 44 to 52 K. For Tc > 100 K the bias grows to -33 K at 2% data and -29 K at 50% masking (single seed).
- **Repeated formulas get one prediction.** `Tl2Ba2Cu1O6` is predicted at 28.5 K for 45 rows with Tc from 10 to 117 K. Single-element `C1` (22 rows, Tc 0.4 to 32 K) is predicted at about 38 K, 5.3% of the grouped squared error.
- **Worst 5% of predictions** are mostly hot cuprates and differ from the rest by 0.6 to 1.6 standard deviations in a few features. This describes them, not a cause.

## Limits

- One dataset and one model. A random forest averages training labels, so it cannot predict above the highest training Tc. Other models were not tested, by design.
- The partition is fixed. Error bars cover model, noise and subsampling randomness, not the partition (item 6).
- Perturbation levels are not commensurable: input noise is train-only, masking is train and test, label noise is not clipped.
- Items 4 to 6 and the failure cases are post-hoc and, where noted, single-run (seed 0).
- Conclusions are specific to this dataset, feature set, baseline, split definitions and perturbation levels.
