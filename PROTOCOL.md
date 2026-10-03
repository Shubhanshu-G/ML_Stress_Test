# PROTOCOL

**Status of this document.** Written 2026-10-03, after the results existed. It describes the protocol recorded in `config.yaml` (commit `a085154`, 2026-09-30 18:10 IST; see `FREEZE.json`). It adds no design choices. Departures and disclosures are listed at the end.

## Question

Assigned scope: a controlled scientific-ML stress test with one public dataset and one fixed baseline model; sweep training-set size, label noise, input noise, missing features and one out-of-distribution split; at least 5 seeds for key comparisons; no leaderboard chasing, model fixed, data interrogated.

Question asked: with the model held fixed, how does test error change as the data changes?

## Hypothesis

No directional hypothesis and no outcome thresholds were written before the run. The protocol specifies what is measured (test RMSE of a fixed model under each data condition), not what result was expected. Consequently no pre-registered outcome failed, and none could. All results are retained, including the large failure on the OOD split.

## Task

Regression: predict critical temperature (Tc, K) of a superconductor from 81 numeric features derived from its chemical formula.

- Dataset: UCI *Superconductivty Data* (id 464), 21,263 rows, files `data/train.csv` and `data/unique_m.csv`, SHA-256 hashes in `config.yaml` and `FREEZE.json`, downloaded 2026-09-30.
- Exclusions: none. All 21,263 rows are used, including 66 exact duplicate rows.

## Controls (held fixed in every run)

- Model: `RandomForestRegressor`, 200 trees, `n_jobs=-1`, `random_state` = run seed. No tuning, no feature engineering, no dimensionality reduction, features used raw.
- Train/test partitions: fixed by `split_seed` in `config.yaml`; seeds change only the model, the noise draws and the subsampling.
- Test labels are never perturbed. Test features are changed only in the missing-feature condition.
- Metric and aggregation: see below.

## Splits

| Split | Definition | Role |
|---|---|---|
| `grouped` | 20% of formulas held out, all rows of a formula on one side (GroupShuffleSplit on the `material` column of `unique_m.csv`) | Primary, swept |
| `ood` | Train on Tc <= 74.0 K (80th percentile of Tc over the full dataset), test on Tc > 74.0 K | Swept |
| `random_iid` | Random 20% of rows | Baseline only (no perturbation) |

## Conditions

One factor at a time from a baseline of (full data, no noise, no masking), plus a size x label-noise grid.

| Factor | Levels | Applied to |
|---|---|---|
| Training fraction | 0.02, 0.05, 0.1, 0.25, 0.5, 1.0 | Training rows |
| Label noise | 0, 0.05, 0.1, 0.2, 0.4 | Training labels; Gaussian, std = level x std of training Tc |
| Input noise | 0, 0.05, 0.1, 0.2, 0.4 | Training features; Gaussian, std = level x training-set std of each feature (raw units; equivalent to std = level on standardized features) |
| Missing features | 0, 0.1, 0.25, 0.5 | Random cells masked in train and test, filled with training means |
| Interaction | fraction {0.1, 0.5, 1.0} x label noise {0, 0.2, 0.4} | As above |

Identical settings are run once and reused (the baseline appears once, and 5 of the 9 grid cells coincide with single-factor runs). Result: 21 conditions per swept split x 2 splits + 1 baseline-only condition = 43 conditions x 10 seeds = 430 runs.

## Metrics

- Primary: test RMSE (K).
- Secondary: MAE and R². R² is not used for the OOD split because its test range is narrow.
- Reference: RMSE of always predicting the training mean.
- Aggregation: mean and sample standard deviation (ddof = 1) over 10 seeds (0 to 9).

## Stopping rule

Run each of the 430 planned runs once. No early stopping on results, no repeats, no changes to seeds, metric, dataset, levels or exclusions after the design commit. The sweep is complete when `results/raw.csv` has 430 unique rows.

## Disclosures and departures

1. **Baseline check before the freeze.** On 2026-09-30 (12:37, before the design commit) an uncommitted script ran the model once on the random split (RMSE 9.43), a grouped split (10.47) and the OOD split (39.19). After seeing that the data contains repeated formulas, `grouped` was added to the design as the primary split. It was in `config.yaml` at the design commit.
2. **Smoke runs after the design commit.** On 2026-09-30 (19:14 to 19:53) `--quick` runs (2 seeds, end levels) tested the pipeline. They led to removing duplicate conditions in `src/sweep.py` (code only). No level, seed, metric or model setting changed. Their output (`results/quick/`) is not retained.
3. **A stopped launch.** On 2026-10-01 (from about 09:29) the sweep was launched under an older commit, before `sweep.py` was committed. It was stopped after about 110 runs and its output deleted, so that every row records the code commit that produced it. The full sweep (10:01 to 11:21) used the same config and the code later identified as `979d3f8`; its values matched the partial output on the rows compared.
4. **Analysis written during the sweep.** `src/analyze.py` was written while the sweep ran and committed after results existed. The audits (`src/audit.py`, `src/audit_errors.py`), `src/claims_check.py` and `src/make_freeze.py` are post-hoc.
5. **Config comment.** `config.yaml` describes input noise as "std on standardized features"; the code uses level x raw feature std. The two are equivalent, and the committed comment was left unchanged.
6. **Documents written after results.** `README.md`, `memo.md` and this file were written after results existed, and the first two were revised after review to match the record.
