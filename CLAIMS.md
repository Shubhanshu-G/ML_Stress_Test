# CLAIMS

Each conclusion in `memo.md` / `README.md` is mapped to the artifact that supports it. Numbers marked **[script]** are recomputed by `python -m src.claims_check` into `results/claims_numbers.csv`. "Post-hoc" means it was found after seeing results. **Unsupported** claims are listed at the end and are not made in the memo.

Artifacts: `results/raw.csv` (430 runs), `results/summary.csv` (mean/std over 10 seeds), `failure_cases/error_by_tc_bin.csv` (seed 0), `results/audit.txt`, `results/audit_errors.txt`, `figures/`.

## A. Supported by the committed sweep

| # | Claim | Evidence |
|---|---|---|
| A1 | Fixed RF baseline RMSE is 9.43 (random), 10.42 (grouped), 39.17 K (OOD); seed std 0.01 / 0.05 / 0.09 | `summary.csv` baseline rows; `figures/baseline_comparison.png` [script] |
| A2 | "Always predict the training mean" scores 34.1 K (grouped), 72.4 K (OOD) | `summary.csv` column `mean_predictor_rmse` [script] |
| A3 | Grouped split, change from baseline at the most-damaged level: 2% data +6.35, 50% masked +5.29, input noise 0.4 +5.20, label noise 0.4 +1.70 K (seed std 0.1 to 0.6) | `summary.csv`; `figures/robustness_curves.png`, `learning_curves.png` [script] |
| A4 | Grouped label noise up to 0.1 changes RMSE by at most +0.13 K (seed std 0.08 to 0.09), i.e. about the size of seed spread | `summary.csv` [script] |
| A5 | Input noise 0.05 already costs +0.78 K on grouped | `summary.csv` |
| A6 | OOD: label noise changes RMSE by at most 0.02 K; input noise by at most 0.69 K (MAE up to +2.4 K) | `summary.csv` [script] |
| A7 | OOD: 50% masked +7.58 K, 2% data +7.06 K | `summary.csv` [script] |
| A8 | Going from 50% to 100% of the training data gains 0.70 K (grouped) and 0.74 K (OOD); the grouped learning curve has not flattened | `summary.csv`; `figures/learning_curves.png` [script] |
| A9 | The same model scores about 4x worse on OOD than on grouped (39.17 vs 10.42 K) | A1 |
| A10 | The sweep is complete: 43 conditions x 10 seeds = 430 rows, one code commit (`979d3f8`) | `FREEZE.json` (`results.complete`, `rows`); `raw.csv` |
| A11 | `config.yaml` is unchanged since its commit `a085154`; dataset hashes match | `FREEZE.json` |
| A12 | The pipeline reproduces: a later `--quick` run matched the committed values to four decimals | 2026-10-02 run log (commit `a9923fd`) |
| A13 | Interaction grid: all 9 cells available (4 `interaction` rows plus 5 shared cells) | `figures/interaction_size_x_labelnoise.png`; `summary.csv` |

## B. Supported, post-hoc (seed 0 unless stated)

| # | Claim | Evidence |
|---|---|---|
| B1 | About 90% (89.6%) of OOD squared error is per-Tc-range bias, not scatter; bias -28.0 K for Tc 74 to 100, -63.8 K above 100 | `error_by_tc_bin.csv` [script]; `figures/pred_vs_true.png` |
| B2 | Grouped baseline error rises with Tc: RMSE 6.3 K (Tc <= 10) to 17.9 K (> 100); bias +1.9 to -9.6 K. Rows above 74 K are 20% of the test set and carry 43% of the squared error | `error_by_tc_bin.csv` [script] |
| B3 | Data reduction and masking push the hottest materials lower (Tc > 100 K bias: -33.4 K at 2% data, -29.5 K at 50% masked, vs -9.6 K at baseline); input noise 0.4 gives -13.2 K and label noise 0.4 -10.0 K | `error_by_tc_bin.csv` [script]. Single seed: the label-noise difference is not meaningful |
| B4 | 37.5% of rows (7,982) belong to a formula that appears more than once; pooled within-formula Tc std is 6.85 K; `Y1Ba2Cu3O7` has 110 rows spanning 30 to 130 K | `audit.txt` |
| B5 | The horizontal streaks in `pred_vs_true.png` are single formulas getting one prediction (`Tl2Ba2Cu1O6`: 28.47 K for 45 rows with Tc 10 to 117 K); 15 of the 25 worst grouped errors are that formula | `audit.txt`; `failure_cases/worst_cases.csv` |
| B6 | One formula (`Tl2Ba2Cu1O6`) causes 16.7% of the committed grouped partition's squared error; RMSE is 9.61 K without it, 9.17 K without the top three formulas | `audit_errors.txt` |
| B7 | Cu+O compounds: 13.6 K RMSE vs 5.6 K for the rest (grouped); already 12.7 K for Tc <= 74 K | `audit_errors.txt` |
| B8 | 841 of the 842 grouped test rows above 74 K are Cu+O compounds; the OOD test set is 99.7% Cu+O (4,237 of 4,249) vs 37.0% of its training rows | `audit.txt`, `audit_errors.txt`, `error_by_tc_bin.csv` |
| B9 | Across ten grouped partitions (split seeds 0 to 9, committed one = seed 0, model seed fixed at 0), baseline RMSE is 9.00 to 10.47 K (mean 9.68, std 0.42); the committed partition is the highest | `audit_errors.txt` (Audit C, also in the run log) |
| B10 | The partition choice moves the baseline (std 0.42) about 8x more than model seed does on one partition (std 0.05) | B9 and A1 |
| B11 | Committed gap, grouped minus random: 0.98 K (10.42 vs 9.43) | `summary.csv` [script] |
| B12 | Feature-shift table: the worst 5% of predictions differ from the rest by about 0.6 to 1.6 std in a few features. Descriptive only | `failure_cases/feature_shift.csv` |

## C. Unsupported or not tested (not claimed as findings)

| # | Statement | Why unsupported |
|---|---|---|
| C1 | The OOD failure is **caused** by missing exposure to the hot Tc range | No experiment varies exposure. The comparison hot cuprates 14.9 K (grouped, hot cuprates in training) vs 39.0 K (OOD) uses different test rows. The memo says "suggests" |
| C2 | The size of the leakage effect (random vs grouped split) | Only 0.98 K on the committed partitions, which are not typical (B9). The random split's own partition spread was not measured. Unresolved |
| C3 | The size of the OOD failure would be different (or the same) for another model | One model only, by design. The "cannot predict above the highest training Tc" statement is a property of random forests, not tested here |
| C4 | The perturbation effects (A3 to A8) would be the same on other test partitions | Only the baseline was repeated across partitions (B9) |
| C5 | Cuprates are hard **because of** a specific chemistry or mechanism | Cuprate status and high Tc are confounded in this dataset (B8). Only the association is shown |
| C6 | Why the worst predictions differ in specific features | Feature shift is descriptive (B12) |
| C7 | A general ranking of noise vs masking vs data size | The levels are not commensurable (masking hits train and test; noise hits train only) |
| C8 | That these results generalize to other datasets, representations or splits | One dataset, one representation, one model |
| C9 | The label-noise penalty at 0.4 grows with data size (+1.04 K at 10%, +1.45 at 50%, +1.70 at 100%, grouped) | Three sizes, no tested explanation. Observed in the grid, **not** claimed in the memo |
