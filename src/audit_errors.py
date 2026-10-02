import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from exception.exception import CustomException
from logger.logger import logger
from src.data import load_config, load_data


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def load_grouped_preds(cfg, tc, g, cuo):
    c = cfg["split"]["grouped"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=c["test_size"],
                                   random_state=c["split_seed"]).split(tc, tc, groups=g))
    p = pd.read_csv("results/preds/grouped__train_fraction_1.0__seed0.csv")
    if len(p) != len(te):
        raise ValueError(f"Prediction file has {len(p)} rows but the grouped test set has {len(te)}")
    p["material"] = g.iloc[te].values
    p["cuo"] = cuo[te]
    p["err"] = p.y_pred - p.y_true
    p["sq"] = p.err ** 2
    return p


def error_concentration(p):
    tot = p.sq.sum()
    t = p.groupby("material").agg(rows=("sq", "size"), pct_sq_error=("sq", lambda s: 100 * s.sum() / tot),
                                  tc_min=("y_true", "min"), tc_max=("y_true", "max"),
                                  mean_true=("y_true", "mean"), mean_pred=("y_pred", "mean"))
    t = t.sort_values("pct_sq_error", ascending=False)
    print(f"grouped baseline, seed 0: rows={len(p)}  RMSE={rmse(p.err):.2f} K")
    print("\nformulas contributing most of the squared error:")
    print(t.head(8).round(1).to_string())
    for k in (1, 3):
        drop = t.index[:k]
        print(f"RMSE without the top {k} formula(s): {rmse(p[~p.material.isin(drop)].err):.2f} K")


def cuprate_split(p):
    print("\ngrouped baseline by composition (Cu+O compounds vs the rest):")
    for flag, grp in p.groupby("cuo"):
        print(f"  Cu+O={flag}: rows={len(grp)} bias={grp.err.mean():.1f} rmse={rmse(grp.err):.1f}")
    cu = p[p.cuo]
    print("within Cu+O compounds, by true Tc:")
    for name, grp in (("Tc <= 74", cu[cu.y_true <= 74]), ("Tc > 74", cu[cu.y_true > 74])):
        print(f"  {name}: rows={len(grp)} bias={grp.err.mean():.1f} rmse={rmse(grp.err):.1f}")


def split_seed_sensitivity(cfg, X, tc, g, n_splits=10):
    c = cfg["split"]["grouped"]
    out = []
    for s in range(n_splits):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=c["test_size"], random_state=s).split(X, tc, groups=g))
        m = RandomForestRegressor(random_state=0, **cfg["model"]["params"]).fit(X.iloc[tr], tc.iloc[tr])
        r = rmse(m.predict(X.iloc[te]) - tc.iloc[te].values)
        out.append(r)
        logger.info(f"split seed {s}: rmse={r:.2f}")
    print(f"\nbaseline RMSE over {n_splits} different grouped test partitions (model seed fixed at 0):")
    print("  " + ", ".join(f"{v:.2f}" for v in out))
    print(f"  mean={np.mean(out):.2f} std={np.std(out, ddof=1):.2f} min={min(out):.2f} max={max(out):.2f}")
    print(f"  committed split_seed={c['split_seed']} gave {out[c['split_seed']]:.2f}" if c["split_seed"] < n_splits else "")


def main():
    try:
        cfg = load_config()
        X, y, groups = load_data(cfg)
        g = groups.reset_index(drop=True)
        tc = y.reset_index(drop=True)
        um = pd.read_csv(cfg["dataset"]["groups_file"])
        cuo = ((um["Cu"] > 0) & (um["O"] > 0)).values

        p = load_grouped_preds(cfg, tc, g, cuo)
        logger.info("Audit A: error concentration by formula")
        error_concentration(p)
        logger.info("Audit B: cuprates vs the rest")
        cuprate_split(p)
        logger.info("Audit C: sensitivity to the test partition (10 refits)")
        split_seed_sensitivity(cfg, X, tc, g)
        logger.info("Audit finished")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()