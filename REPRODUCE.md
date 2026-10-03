# REPRODUCE

Python and library versions: see `FREEZE.json` (`model.python`, `model.libraries`) and `requirements.txt`.

## 1. Setup (about 2 minutes)

```
git clone <repo url>
cd <repo folder>
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Regenerate figures and tables from the committed results (under 1 minute)

```
python run.py
```

`results/raw.csv` is committed with all 430 runs, so the sweep step finds them done and only the analysis runs. Outputs: `figures/`, `results/summary.csv`, `failure_cases/`.

## 3. Recompute every run from scratch (about 80 minutes on a laptop CPU)

```
python run.py --fresh
```

This deletes `results/raw.csv` and `results/preds/`, then reruns all 430 fits and the analysis. Check the new metrics against the committed ones before committing anything:

```
python -c "import io, subprocess, pandas as pd; old=pd.read_csv(io.StringIO(subprocess.check_output(['git','show','HEAD:results/raw.csv']).decode())); new=pd.read_csv('results/raw.csv'); k=['factor','level','split','seed']; m=old.merge(new,on=k,suffixes=('_o','_n')); print(len(old),len(new),len(m)); print({c: float((m[c+'_o']-m[c+'_n']).abs().max()) for c in ['rmse','mae','r2']})"
```

Expected: `430 430 430` and maximum differences of `0.0` with the same library versions. The `timestamp` and `git_hash` columns will differ. To restore the committed results afterwards: `git checkout -- results`.

## 4. Quick smoke test (about 5 minutes)

```
python run.py --quick
```

2 seeds and the end levels of each factor; writes to `results/quick/` (not committed). A run on 2026-10-02 reproduced the committed values to four decimals.

## 5. Verify the freeze

```
git show a085154:config.yaml
git diff a085154 HEAD -- config.yaml
python -m src.make_freeze
```

The `git diff` prints nothing, and `make_freeze` prints `config_unchanged_since_design_commit: true`, both dataset hashes matching, and `results complete: True (430 rows)`.

## 6. Post-hoc scripts (not part of `run.py`)

```
python -m src.audit > results/audit.txt
python -m src.audit_errors > results/audit_errors.txt
python -m src.claims_check
```

`claims_check` recomputes the derived numbers quoted in `README.md` and `memo.md` into `results/claims_numbers.csv`. Which claim rests on which file: `CLAIMS.md`.
