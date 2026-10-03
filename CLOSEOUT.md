# CLOSEOUT

**Status: MIXED**

**Reason.** The planned experiment ran to completion: all 430 planned runs (43 conditions x 10 seeds) are retained, with no condition dropped and nothing blocked (one earlier partial launch was discarded and rerun with the same config; see `PROTOCOL.md`). The retained result is mixed. A fixed Random Forest on UCI Superconductivity was robust to label noise (and, on the OOD split, to input noise), but its error rose sharply with a distribution shift (test RMSE 39.2 K on the OOD split against 10.4 K on the grouped split) and with heavy feature masking or training-data reduction.

No hypothesis or outcome thresholds were written before the run, so no pre-registered outcome failed.

**Main limits.** One dataset and one model. The test partition is fixed, and the committed grouped partition is the highest of ten partitions tried, itself included (9.00 to 10.47 K), so absolute levels depend on it. Perturbation levels are not comparable across factors. Tc and chemistry (copper-oxide compounds) are confounded in the hot range. Findings from the data audits are post-hoc and, where noted, single-seed. Explanations of the OOD failure are suggestions and not tested; see `CLAIMS.md`.

**Where to look.** Protocol: `PROTOCOL.md`. Freeze: `FREEZE.json` (design commit `a085154`, code commit `979d3f8`). Results: `results/raw.csv`. Claims and their evidence: `CLAIMS.md`. Reproduction: `REPRODUCE.md`. Write-up: `memo.md`.
