import numpy as np
import pandas as pd

### seed give same value after every run

def _rng(seed, code):  #Random number genenrator, 
    """Separate random stream per (seed, perturbation type)."""
    return np.random.default_rng([seed, code])


def subsample(X, y, frac, seed):
    """Keep a random fraction of the training rows."""
    if frac >= 1.0:
        return X.copy(), y.copy()
    n = int(round(len(X) * frac))
    idx = _rng(seed, 1).choice(len(X), size=n, replace=False) # replace = False dont selct same row again
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def add_label_noise(y, level, seed): # Add noise to target
    """Add gaussian noise to training labels. std = level * std(y). Train only."""
    if level == 0:
        return y.copy()
    noise = _rng(seed, 2).normal(0, level * y.std(), size=len(y))
    return y + noise


def add_input_noise(X, level, seed): # Add noise to inputs
    """Add gaussian noise to each feature. std = level * std of that feature. Train only."""
    if level == 0:
        return X.copy()
    noise = _rng(seed, 3).normal(0, 1, size=X.shape) * (level * X.std().values)
    return X + noise


def mask_features(X_train, X_test, frac, seed):
    """Randomly mask a fraction of cells in train AND test.
    Masked cells are filled with the TRAIN column mean (computed before masking)."""
    if frac == 0:
        return X_train.copy(), X_test.copy()
    fill = X_train.mean().values
    rng = _rng(seed, 4)
    tr = X_train.to_numpy(dtype=float, copy=True)
    te = X_test.to_numpy(dtype=float, copy=True)
    tr = np.where(rng.random(tr.shape) < frac, fill, tr)
    te = np.where(rng.random(te.shape) < frac, fill, te)
    return (pd.DataFrame(tr, columns=X_train.columns),
            pd.DataFrame(te, columns=X_test.columns))


if __name__ == "__main__":
    from src.data import load_config, load_data, get_split

    cfg = load_config()
    X, y, groups = load_data(cfg)
    Xtr, Xte, ytr, yte = get_split("grouped", X, y, groups, cfg)
    y_before = ytr.copy()

    Xs, ys = subsample(Xtr, ytr, 0.1, seed=0)
    print("subsample 0.1:", len(Xs), "of", len(Xtr))

    yn = add_label_noise(ytr, 0.4, seed=0)
    print("label noise std ratio (expect ~0.4):", round((yn - ytr).std() / ytr.std(), 3))

    Xn = add_input_noise(Xtr, 0.2, seed=0)
    print("input noise std ratio (expect ~0.2):", round(((Xn - Xtr).std() / Xtr.std()).mean(), 3))

    Mtr, Mte = mask_features(Xtr, Xte, 0.25, seed=0)
    print("masked share train (expect ~0.25):", round((Mtr.values != Xtr.values).mean(), 3))
    print("masked share test  (expect ~0.25):", round((Mte.values != Xte.values).mean(), 3))
    print("NaNs after masking (expect 0):", int(Mtr.isna().sum().sum() + Mte.isna().sum().sum()))

    a = add_label_noise(ytr, 0.2, seed=1)
    b = add_label_noise(ytr, 0.2, seed=1)
    c = add_label_noise(ytr, 0.2, seed=2)
    print("same seed identical:", np.allclose(a, b), "| different seed differs:", not np.allclose(a, c))
    print("original labels untouched:", y_before.equals(ytr))
    logging.info(
    f"subsample 0.1: {len(Xs)} of {len(Xtr)}\n"
    f"label noise std ratio (expect ~0.4): {round((yn - ytr).std() / ytr.std(), 3)}\n"
    f"input noise std ratio (expect ~0.2): {round(((Xn - Xtr).std() / Xtr.std()).mean(), 3)}\n"
    f"masked share train (expect ~0.25): {round((Mtr.values != Xtr.values).mean(), 3)}\n"
    f"masked share test  (expect ~0.25): {round((Mte.values != Xte.values).mean(), 3)}\n"
    f"NaNs after masking (expect 0): {int(Mtr.isna().sum().sum() + Mte.isna().sum().sum())}\n"
    f"same seed identical: {np.allclose(a, b)} | different seed differs: {not np.allclose(a, c)}\n"
    f"original labels untouched: {y_before.equals(ytr)}"
    )