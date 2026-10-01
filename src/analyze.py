import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exception.exception import CustomException
from logger.logger import logger
from src.data import get_split, load_config, load_data

SETTINGS = ["train_fraction", "label_noise", "input_noise", "missing_frac"]
BASE = dict(train_fraction=1.0, label_noise=0.0, input_noise=0.0, missing_frac=0.0)
COLORS = {"grouped": "tab:blue", "ood": "tab:red", "random_iid": "tab:gray"}
TC_BINS = [-np.inf, 10, 20, 40, 74, 100, np.inf]
TC_LABELS = ["<=10", "10-20", "20-40", "40-74", "74-100", ">100"]


def select(df, split, vary):
    """Rows of one split where every setting except `vary` is at its baseline value."""
    d = df[df["split"] == split]
    for k, v in BASE.items():
        if k != vary:
            d = d[np.isclose(d[k], v)]
    return d


def curve(d, x):
    g = d.groupby(x)["rmse"].agg(["mean", "std", "count"]).reset_index().sort_values(x)
    g["std"] = g["std"].fillna(0)
    return g


def ref_rmse(df, split):
    return df[df["split"] == split]["mean_predictor_rmse"].mean()


# ---------- figures ----------

def fig_learning(df, fdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ["grouped", "ood"]):
        g = curve(select(df, split, "train_fraction"), "train_fraction")
        ax.errorbar(g["train_fraction"], g["mean"], yerr=g["std"], marker="o",
                    capsize=3, color=COLORS[split], label="RF, mean +/- std over seeds")
        ax.axhline(ref_rmse(df, split), ls="--", color="k", lw=1, label="always predict train mean")
        ax.set_xscale("log")
        ax.set_xlabel("training-set fraction")
        ax.set_ylabel("test RMSE (K)")
        ax.set_title(f"Learning curve: {split}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "learning_curves.png"), dpi=150)
    plt.close(fig)


def fig_robustness(df, fdir):
    factors = [("label_noise", "label noise (x std of Tc)"),
               ("input_noise", "input noise (x std of each feature)"),
               ("missing_frac", "fraction of features masked")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (f, label) in zip(axes, factors):
        for split in ["grouped", "ood"]:
            g = curve(select(df, split, f), f)
            ax.errorbar(g[f], g["mean"], yerr=g["std"], marker="o", capsize=3,
                        color=COLORS[split], label=split)
        ax.set_xlabel(label)
        ax.set_ylabel("test RMSE (K)")
        ax.set_title(f"Robustness: {f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "robustness_curves.png"), dpi=150)
    plt.close(fig)


def fig_baselines(df, fdir):
    names, means, stds, refs = [], [], [], []
    for split in ["random_iid", "grouped", "ood"]:
        d = df[df["split"] == split]
        for k, v in BASE.items():
            d = d[np.isclose(d[k], v)]
        if d.empty:
            continue
        names.append(split)
        means.append(d["rmse"].mean())
        stds.append(d["rmse"].std() if len(d) > 1 else 0)
        refs.append(d["mean_predictor_rmse"].mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=stds, capsize=4, color=[COLORS[n] for n in names])
    ax.scatter(names, refs, color="k", marker="_", s=600, label="always predict train mean")
    ax.set_ylabel("test RMSE (K)")
    ax.set_title("Same model, three tests (no perturbation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "baseline_comparison.png"), dpi=150)
    plt.close(fig)


def fig_interaction(df, cfg, fdir):
    inter = cfg["sweeps"]["interaction"]
    tfs, lns = inter["train_fraction"], inter["label_noise"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ["grouped", "ood"]):
        d = df[(df["split"] == split) & np.isclose(df["input_noise"], 0) & np.isclose(df["missing_frac"], 0)]
        m = d.pivot_table(index="train_fraction", columns="label_noise", values="rmse", aggfunc="mean").reindex(index=tfs, columns=lns)
        s = d.pivot_table(index="train_fraction", columns="label_noise", values="rmse", aggfunc="std").reindex(index=tfs, columns=lns)
        ax.imshow(m.values.astype(float), cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(lns)))
        ax.set_xticklabels(lns)
        ax.set_yticks(range(len(tfs)))
        ax.set_yticklabels(tfs)
        ax.set_xlabel("label noise")
        ax.set_ylabel("training fraction")
        ax.set_title(f"RMSE (K), {split}")
        for i in range(len(tfs)):
            for j in range(len(lns)):
                v = m.iloc[i, j]
                txt = "n/a" if pd.isna(v) else f"{v:.1f}\n+/-{0 if pd.isna(s.iloc[i, j]) else s.iloc[i, j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "interaction_size_x_labelnoise.png"), dpi=150)
    plt.close(fig)


def fig_scatter(preds, fdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split in zip(axes, ["grouped", "ood"]):
        p = preds.get(f"{split}__train_fraction_1.0__seed0")
        if p is None:
            ax.set_visible(False)
            continue
        ax.scatter(p["y_true"], p["y_pred"], s=4, alpha=0.3, color=COLORS[split])
        lim = [0, max(p["y_true"].max(), p["y_pred"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("true Tc (K)")
        ax.set_ylabel("predicted Tc (K)")
        ax.set_title(f"Predicted vs true, {split} (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "pred_vs_true.png"), dpi=150)
    plt.close(fig)


# ---------- tables ----------

def make_summary(df, rdir):
    g = df.groupby(["split"] + SETTINGS)
    s = g.agg(n_seeds=("seed", "nunique"), n_train=("n_train", "mean"),
              rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
              mae_mean=("mae", "mean"), mae_std=("mae", "std"),
              r2_mean=("r2", "mean"), r2_std=("r2", "std"),
              mean_predictor_rmse=("mean_predictor_rmse", "mean")).reset_index()
    s["n_train"] = s["n_train"].round(0)
    s.round(4).to_csv(os.path.join(rdir, "summary.csv"), index=False)
    return s


# ---------- failure cases ----------

def worst_rows(Xte, p, name, top=25):
    d = p[["y_true", "y_pred"]].copy()
    d["error"] = d["y_pred"] - d["y_true"]
    d["abs_error"] = d["error"].abs()
    d = pd.concat([d, Xte.reset_index(drop=True)], axis=1)
    d = d.sort_values("abs_error", ascending=False).head(top)
    d.insert(0, "condition", name)
    return d


def error_by_bin(p, name):
    d = pd.DataFrame({"tc_bin": pd.cut(p["y_true"], bins=TC_BINS, labels=TC_LABELS),
                      "err": p["y_pred"] - p["y_true"]})
    t = d.groupby("tc_bin", observed=True).agg(
        n=("err", "size"), bias=("err", "mean"),
        rmse=("err", lambda e: float(np.sqrt((e ** 2).mean())))).reset_index()
    t.insert(0, "condition", name)
    return t


def feature_shift(Xte, p, name, top=10):
    """Which features differ most between the worst 5% of predictions and the rest."""
    ae = (p["y_pred"] - p["y_true"]).abs()
    worst = (ae >= ae.quantile(0.95)).values
    sd = Xte.std().replace(0, np.nan)
    z = ((Xte[worst].mean() - Xte[~worst].mean()) / sd).dropna()
    z = z.reindex(z.abs().sort_values(ascending=False).index).head(top)
    out = z.rename("std_diff").reset_index().rename(columns={"index": "feature"})
    out.insert(0, "condition", name)
    return out


def failure_cases(cfg, rdir, cdir):
    X, y, groups = load_data(cfg)
    splits = {s: get_split(s, X, y, groups, cfg) for s in ["random_iid", "grouped", "ood"]}
    preds, worst, bins, shifts = {}, [], [], []
    for path in sorted(glob.glob(os.path.join(rdir, "preds", "*.csv"))):
        name = os.path.basename(path)[:-4]
        split = name.split("__")[0]
        p = pd.read_csv(path)
        Xte = splits[split][1]
        if len(p) != len(Xte):
            logger.warning(f"Skipping {name}: prediction length does not match test set")
            continue
        preds[name] = p
        worst.append(worst_rows(Xte, p, name))
        bins.append(error_by_bin(p, name))
        shifts.append(feature_shift(Xte, p, name))
    if worst:
        pd.concat(worst).round(4).to_csv(os.path.join(cdir, "worst_cases.csv"), index=False)
        pd.concat(bins).round(3).to_csv(os.path.join(cdir, "error_by_tc_bin.csv"), index=False)
        pd.concat(shifts).round(3).to_csv(os.path.join(cdir, "feature_shift.csv"), index=False)
    return preds


def main(quick=False):
    try:
        cfg = load_config()
        rdir = "results/quick" if quick else "results"
        fdir = "figures/quick" if quick else "figures"
        cdir = "failure_cases/quick" if quick else "failure_cases"
        for d in (fdir, cdir):
            os.makedirs(d, exist_ok=True)

        df = pd.read_csv(os.path.join(rdir, "raw.csv"))
        seeds_per_cell = df.groupby(["split"] + SETTINGS)["seed"].nunique()
        logger.info(f"Loaded {len(df)} rows; seeds per condition: min {seeds_per_cell.min()}")
        if not quick and seeds_per_cell.min() < 5:
            logger.warning("Some conditions have fewer than 5 seeds. Has the sweep finished?")

        fig_learning(df, fdir)
        fig_robustness(df, fdir)
        fig_baselines(df, fdir)
        fig_interaction(df, cfg, fdir)
        make_summary(df, rdir)
        preds = failure_cases(cfg, rdir, cdir)
        fig_scatter(preds, fdir)
        logger.info(f"Analysis done. Figures in {fdir}, failure cases in {cdir}")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)