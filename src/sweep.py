import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from exception.exception import CustomException
from logger.logger import logger
from src.data import get_split, load_config, load_data
from src.perturb import add_input_noise, add_label_noise, mask_features, subsample

COLUMNS = [
    "factor", "level", "split", "seed", "train_fraction", "label_noise",
    "input_noise", "missing_frac", "n_train", "rmse", "mae", "r2",
    "mean_predictor_rmse", "git_hash", "timestamp",
]
FACTORS = ["train_fraction", "label_noise", "input_noise", "missing_frac"]


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_conditions(cfg, quick=False):
    """One-factor-at-a-time conditions plus the interaction grid, with duplicates removed."""
    sw = cfg["sweeps"]
    pick = (lambda lv: [lv[0], lv[-1]]) if quick else (lambda lv: list(lv))
    base = dict(train_fraction=1.0, label_noise=0, input_noise=0, missing_frac=0)

    conds = []
    for factor in FACTORS:
        levels = sw[factor]
        for lv in pick(levels):
            c = dict(base)
            c[factor] = lv
            c.update(factor=factor, level=lv,
                     keep_preds=lv in (levels[0], levels[-1]))   # baseline and most-damaged end
            conds.append(c)

    inter = sw["interaction"]
    combos = [(f, n) for f in inter["train_fraction"] for n in inter["label_noise"]]
    if quick:
        combos = [combos[0], combos[-1]]
    for f, n in combos:
        c = dict(base)
        c.update(train_fraction=f, label_noise=n, factor="interaction",
                 level=f"tf={f}|ln={n}", keep_preds=False)
        conds.append(c)

    # drop conditions whose four settings were already seen (results would be identical)
    unique, seen = [], set()
    for c in conds:
        k = (c["train_fraction"], c["label_noise"], c["input_noise"], c["missing_frac"])
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique, base


def load_done(csv_path):
    """Keys of runs already in the CSV, so an interrupted sweep can resume."""
    if not os.path.exists(csv_path):
        return set()
    d = pd.read_csv(csv_path, dtype=str)
    return {(r.split, r.factor, r.level, int(r.seed)) for r in d.itertuples()}


def append_row(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(splits, cfg, split, c, seed):
    Xtr, Xte, ytr, yte = splits[split]

    # damage the TRAIN data only (noise first, on the full train set, then subsample)
    yt = add_label_noise(ytr, c["label_noise"], seed)
    Xt = add_input_noise(Xtr, c["input_noise"], seed)
    Xt, yt = subsample(Xt, yt, c["train_fraction"], seed)
    Xt, Xe = mask_features(Xt, Xte, c["missing_frac"], seed)   # masks train AND test

    model = RandomForestRegressor(random_state=seed, **cfg["model"]["params"])
    model.fit(Xt, yt)
    pred = model.predict(Xe)

    rmse = mean_squared_error(yte, pred) ** 0.5
    ref = mean_squared_error(yte, np.full(len(yte), ytr.mean())) ** 0.5   # "always guess the mean"
    row = {
        "factor": c["factor"], "level": c["level"], "split": split, "seed": seed,
        "train_fraction": c["train_fraction"], "label_noise": c["label_noise"],
        "input_noise": c["input_noise"], "missing_frac": c["missing_frac"],
        "n_train": len(Xt), "rmse": round(rmse, 4),
        "mae": round(mean_absolute_error(yte, pred), 4),
        "r2": round(r2_score(yte, pred), 4),
        "mean_predictor_rmse": round(ref, 4),
    }
    return row, pred


def save_preds(out_dir, split, c, yte, pred):
    level = str(c["level"])
    name = f"{split}__{c['factor']}_{level}__seed0.csv"
    df = pd.DataFrame({"test_row": np.arange(len(yte)), "y_true": yte.values, "y_pred": pred})
    df.round(3).to_csv(os.path.join(out_dir, "preds", name), index=False)


def run_sweep(cfg, quick=False):
    try:
        out_dir = "results/quick" if quick else "results"
        os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
        csv_path = os.path.join(out_dir, "raw.csv")
        seeds = cfg["seeds"][:2] if quick else cfg["seeds"]
        git_hash = get_git_hash()

        X, y, groups = load_data(cfg)
        all_splits = cfg["sweep_splits"] + cfg["baseline_only_splits"]
        splits = {s: get_split(s, X, y, groups, cfg) for s in all_splits}

        conds, base = make_conditions(cfg, quick)
        jobs = [(s, c) for s in cfg["sweep_splits"] for c in conds]
        for s in cfg["baseline_only_splits"]:
            jobs.append((s, dict(base, factor="baseline", level=0, keep_preds=True)))

        done = load_done(csv_path)
        total = len(jobs) * len(seeds)
        n = 0
        logger.info(f"Sweep start: {total} runs, git {git_hash}, output {csv_path}")

        for split, c in jobs:
            for seed in seeds:
                n += 1
                key = (split, c["factor"], str(c["level"]), int(seed))
                if key in done:
                    continue
                row, pred = run_one(splits, cfg, split, c, seed)
                row["git_hash"] = git_hash
                row["timestamp"] = datetime.now().isoformat(timespec="seconds")
                append_row(csv_path, row)                  # write the result first
                if seed == 0 and c["keep_preds"]:
                    save_preds(out_dir, split, c, splits[split][3], pred)
                logger.info(f"[{n}/{total}] {split} {c['factor']}={c['level']} seed={seed} "
                            f"rmse={row['rmse']}")
        logger.info("Sweep finished")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the run count and exit")
    args = ap.parse_args()
    cfg = load_config()
    if args.dry_run:
        conds, _ = make_conditions(cfg, args.quick)
        seeds = cfg["seeds"][:2] if args.quick else cfg["seeds"]
        n = (len(conds) * len(cfg["sweep_splits"]) + len(cfg["baseline_only_splits"])) * len(seeds)
        print(f"{len(conds)} conditions per split, {n} runs total")
    else:
        run_sweep(cfg, quick=args.quick)