"""data loading + train/test splitting module"""
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from exception.exception import CustomException
from logger.logger import logger


def load_config(path="config.yaml"):  #Taking numerical values from yaml files
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    """Returns features X, target y, and group ids (material name) per row."""
    try:
        df = pd.read_csv(cfg["dataset"]["source_file"])
        groups = pd.read_csv(cfg["dataset"]["groups_file"])["material"]
        target = cfg["dataset"]["target"]
        X = df.drop(columns=target)
        y = df[target]
        logger.info(f"Loaded data: {X.shape[0]} rows, {X.shape[1]} features")
        return X, y, groups
    except Exception as e:
        raise CustomException(e, sys)


def _pack(X, y, train_idx, test_idx):
    """Cut X and y by row position and reset the index so later steps stay simple."""
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def get_split(name, X, y, groups, cfg):
    """name: 'random_iid', 'grouped' or 'ood'. Returns X_train, X_test, y_train, y_test."""
    try:
        idx = np.arange(len(X))

        if name == "random_iid":
            c = cfg["split"]["random_iid"]
            tr, te = train_test_split(idx, test_size=c["test_size"], random_state=c["split_seed"])  #For random or IID split

        elif name == "grouped":
            c = cfg["split"]["grouped"]
            gss = GroupShuffleSplit(n_splits=1, test_size=c["test_size"], random_state=c["split_seed"])  #For group split
            tr, te = next(gss.split(X, y, groups=groups))

        elif name == "ood":
            thr = y.quantile(cfg["split"]["ood"]["quantile"]) #Out Of Distribution split
            tr = idx[(y <= thr).values]
            te = idx[(y > thr).values]

        else:
            raise ValueError(f"Unknown split: {name}")

        return _pack(X, y, tr, te)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    cfg = load_config()
    X, y, groups = load_data(cfg)
    g = groups.reset_index(drop=True)

    for name in ["random_iid", "grouped", "ood"]:
        Xtr, Xte, ytr, yte = get_split(name, X, y, groups, cfg)
        print(f"{name:10s} train={len(Xtr)} test={len(Xte)} "
              f"Tc train max={ytr.max():.1f} test min={yte.min():.1f}")

    # check: no material appears on both sides of the grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["grouped"]["test_size"],
                            random_state=cfg["split"]["grouped"]["split_seed"])
    tr, te = next(gss.split(X, y, groups=groups))
    overlap = set(groups.iloc[tr]) & set(groups.iloc[te])
    print("grouped overlap (must be 0):", len(overlap))