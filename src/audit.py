import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from exception.exception import CustomException
from logger.logger import logger
from src.data import load_config, load_data


def repeated_formulas(g, tc):
    """How many rows share a formula, and how much Tc varies inside one formula."""
    size = g.map(g.value_counts())
    rep = (size > 1).values
    print(f"rows={len(tc)}  formulas={g.nunique()}  "
          f"rows in repeated formulas={rep.sum()} ({rep.mean():.1%})")

    d = pd.DataFrame({"material": g, "tc": tc})
    agg = d.groupby("material").tc.agg(count="size", tc_min="min", tc_max="max")
    agg["range"] = agg.tc_max - agg.tc_min
    rep_agg = agg[agg["count"] > 1]

    dr = d[rep]
    dev = dr.tc - dr.groupby("material").tc.transform("mean")
    print(f"pooled within-formula std (repeated formulas only): {np.sqrt((dev ** 2).mean()):.2f} K")
    print(f"median Tc range inside a repeated formula: {rep_agg['range'].median():.1f} K; "
          f"share with range > 10 K: {(rep_agg['range'] > 10).mean():.1%}")
    print("\nrepeated formulas with the widest Tc range:")
    print(rep_agg.sort_values("range", ascending=False).head(8).round(1).to_string())


def grouped_streaks(cfg, X, tc, g):
    """Who is behind the repeated predictions on the grouped test set (seed 0)?"""
    c = cfg["split"]["grouped"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=c["test_size"],
                                   random_state=c["split_seed"]).split(X, tc, groups=g))
    p = pd.read_csv("results/preds/grouped__train_fraction_1.0__seed0.csv")
    if len(p) != len(te):
        raise ValueError(f"Prediction file has {len(p)} rows but the grouped test set has {len(te)}")
    p["material"] = g.iloc[te].values
    p["abs_err"] = (p.y_pred - p.y_true).abs()

    print("\nmost repeated predictions on the grouped test set:")
    s = p.groupby(p.y_pred.round(2)).agg(rows=("y_true", "size"),
                                          materials=("material", "nunique"),
                                          tc_min=("y_true", "min"),
                                          tc_max=("y_true", "max"))
    print(s.sort_values("rows", ascending=False).head(5).to_string())

    print("\nmaterials among the 25 worst grouped errors:")
    print(p.sort_values("abs_err", ascending=False).head(25).material.value_counts().head(8).to_string())


def ood_composition(cfg, tc):
    """What kind of material sits in the OOD test set?"""
    um = pd.read_csv(cfg["dataset"]["groups_file"])
    if not {"Cu", "O"} <= set(um.columns):
        logger.warning("Cu/O columns not found in the groups file; skipping composition check")
        print("\nCu/O columns not found; first columns:", list(um.columns)[:10])
        return

    cuo = ((um["Cu"] > 0) & (um["O"] > 0)).values
    thr = tc.quantile(cfg["split"]["ood"]["quantile"])
    test_mask = (tc > thr).values
    idx = np.where(test_mask)[0]

    po = pd.read_csv("results/preds/ood__train_fraction_1.0__seed0.csv")
    if len(po) != len(idx):
        raise ValueError(f"Prediction file has {len(po)} rows but the OOD test set has {len(idx)}")
    po["cuo"] = cuo[idx]
    po["err"] = po.y_pred - po.y_true

    print(f"\nCu+O compounds: {cuo[~test_mask].mean():.1%} of OOD train rows, "
          f"{cuo[test_mask].mean():.1%} of OOD test rows")
    for flag, grp in po.groupby("cuo"):
        print(f"OOD test, Cu+O={flag}: rows={len(grp)} bias={grp.err.mean():.1f} "
              f"rmse={np.sqrt((grp.err ** 2).mean()):.1f}")


def main():
    try:
        cfg = load_config()
        X, y, groups = load_data(cfg)
        g = groups.reset_index(drop=True)
        tc = y.reset_index(drop=True)

        logger.info("Audit 1/3: repeated formulas")
        repeated_formulas(g, tc)

        logger.info("Audit 2/3: grouped test set predictions")
        grouped_streaks(cfg, X, tc, g)

        logger.info("Audit 3/3: OOD test set composition")
        ood_composition(cfg, tc)

        logger.info("Audit finished")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()