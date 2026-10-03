"""Recompute the derived numbers quoted in README.md / memo.md from the result files.
Post-hoc. Writes results/claims_numbers.csv. Run from the project root: python -m src.claims_check
Reads: results/summary.csv, failure_cases/error_by_tc_bin.csv. Audit numbers come from results/audit*.txt."""
import sys

import numpy as np
import pandas as pd

from exception.exception import CustomException
from logger.logger import logger

BASE = dict(train_fraction=1.0, label_noise=0.0, input_noise=0.0, missing_frac=0.0)
rows = []


def add(claim, value, note=""):
    rows.append({"claim": claim, "value": value, "note": note})
    print(f"{claim}: {value}  {note}")


def pick(s, split, **kw):
    cond = {**BASE, **kw}
    d = s[s["split"] == split]
    for k, v in cond.items():
        d = d[np.isclose(d[k], v)]
    if len(d) != 1:
        raise ValueError(f"expected 1 summary row for {split} {kw}, found {len(d)}")
    return d.iloc[0]


def bins(b, cond):
    d = b[b["condition"] == cond]
    if d.empty:
        raise ValueError(f"no rows for {cond} in error_by_tc_bin.csv")
    return d.set_index("tc_bin")


def main():
    try:
        s = pd.read_csv("results/summary.csv")
        b = pd.read_csv("failure_cases/error_by_tc_bin.csv")

        # 1. baselines and reference
        base = {sp: pick(s, sp) for sp in ("random_iid", "grouped", "ood")}
        for sp, r in base.items():
            add(f"baseline RMSE, {sp}", round(r.rmse_mean, 2), f"seed std {r.rmse_std:.2f}, n_seeds {int(r.n_seeds)}")
            add(f"always-predict-mean RMSE, {sp}", round(r.mean_predictor_rmse, 1))
        add("committed gap, grouped minus random", round(base["grouped"].rmse_mean - base["random_iid"].rmse_mean, 2),
            "committed partitions only")

        # 2. change from baseline for each perturbation
        tests = {
            "training size 50%": dict(train_fraction=0.5),
            "training size 10%": dict(train_fraction=0.1),
            "training size 2%": dict(train_fraction=0.02),
            "label noise 0.1": dict(label_noise=0.1),
            "label noise 0.4": dict(label_noise=0.4),
            "input noise 0.2": dict(input_noise=0.2),
            "input noise 0.4": dict(input_noise=0.4),
            "10% masked": dict(missing_frac=0.1),
            "50% masked": dict(missing_frac=0.5),
        }
        for sp in ("grouped", "ood"):
            for name, kw in tests.items():
                r = pick(s, sp, **kw)
                add(f"{sp}: {name}", round(r.rmse_mean, 2),
                    f"change vs baseline {r.rmse_mean - base[sp].rmse_mean:+.2f} K, seed std {r.rmse_std:.2f}")
        r50 = pick(s, "ood", train_fraction=0.5)
        add("ood: gain from 50% to 100% of the data", round(r50.rmse_mean - base["ood"].rmse_mean, 2))

        # 3. bias vs scatter and per-range bias (seed 0, error_by_tc_bin.csv)
        o = bins(b, "ood__train_fraction_1.0__seed0")
        share = (o["n"] * o["bias"] ** 2).sum() / (o["n"] * o["rmse"] ** 2).sum()
        add("ood baseline: share of squared error that is per-range bias", f"{share:.1%}", "seed 0")
        for k in ("74-100", ">100"):
            add(f"ood baseline bias, Tc {k}", round(o.loc[k, "bias"], 1), "K, seed 0")
        g = bins(b, "grouped__train_fraction_1.0__seed0")
        add("grouped baseline bias, Tc <= 10", round(g.loc["<=10", "bias"], 1), "K, seed 0")
        add("grouped baseline bias, Tc > 100", round(g.loc[">100", "bias"], 1), "K, seed 0")
        add("grouped test rows with Tc > 74", int(g.loc["74-100", "n"] + g.loc[">100", "n"]), "seed 0 test set")
        for cond, label in (("grouped__train_fraction_0.02__seed0", "2% data"),
                            ("grouped__missing_frac_0.5__seed0", "50% masked"),
                            ("grouped__input_noise_0.4__seed0", "input noise 0.4"),
                            ("grouped__label_noise_0.4__seed0", "label noise 0.4")):
            add(f"grouped bias, Tc > 100, {label}", round(bins(b, cond).loc[">100", "bias"], 1), "K, seed 0")

        pd.DataFrame(rows).to_csv("results/claims_numbers.csv", index=False)
        logger.info(f"Wrote results/claims_numbers.csv ({len(rows)} rows)")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
