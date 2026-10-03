"""Write FREEZE.json: the freeze manifest (commit SHAs, dataset hashes, model and library versions).
Run once from the project root: python -m src.make_freeze
Generated after the results existed; it records facts from git history and the committed files."""
import hashlib
import json
import platform
import subprocess
import sys

import matplotlib
import numpy
import pandas
import sklearn
import yaml

from exception.exception import CustomException
from logger.logger import logger
from src.data import load_config

DESIGN_COMMIT = "a085154"   # commit that added config.yaml (the design)
CODE_COMMIT = "979d3f8"     # commit of the sweep code that produced results/raw.csv
PLANNED_RUNS = 430          # 43 conditions x 10 seeds


def git(*args):
    return subprocess.check_output(["git", *args]).decode().strip()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def commit_info(short, contains):
    return {
        "sha": git("rev-parse", short),
        "time": git("log", "-1", "--format=%aI", short),
        "message": git("log", "-1", "--format=%s", short),
        "contains": contains,
    }


def main():
    try:
        cfg = load_config()
        d = cfg["dataset"]
        with open("results/raw.csv") as f:
            n_rows = sum(1 for _ in f) - 1

        manifest = {
            "note": ("Freeze manifest generated after the results existed, from git history and the committed "
                     "files. The authoritative pre-result record is config.yaml at design_commit."),
            "design_commit": commit_info(DESIGN_COMMIT, "config.yaml (splits, levels, seeds, model settings, metric)"),
            "code_commit": commit_info(CODE_COMMIT, "sweep code that produced results/raw.csv"),
            "config_unchanged_since_design_commit": git("diff", DESIGN_COMMIT, "HEAD", "--", "config.yaml") == "",
            "config_sha256_now": sha256("config.yaml"),
            "dataset": {
                "name": d["name"],
                "uci_id": d["uci_id"],
                "download_date": str(d["download_date"]),
                "train_csv": {"path": d["source_file"], "sha256_in_config": d["sha256_train"],
                              "sha256_now": sha256(d["source_file"]),
                              "match": d["sha256_train"] == sha256(d["source_file"])},
                "unique_m_csv": {"path": d["groups_file"], "sha256_in_config": d["sha256_unique_m"],
                                 "sha256_now": sha256(d["groups_file"]),
                                 "match": d["sha256_unique_m"] == sha256(d["groups_file"])},
                "target": d["target"],
            },
            "model": {
                "type": cfg["model"]["type"],
                "params": cfg["model"]["params"],
                "random_state": "per-run seed",
                "python": platform.python_version(),
                "libraries": {
                    "numpy": numpy.__version__, "pandas": pandas.__version__,
                    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                    "pyyaml": yaml.__version__,
                },
            },
            "design": {
                "primary_split": cfg["primary_split"],
                "sweep_splits": cfg["sweep_splits"],
                "baseline_only_splits": cfg["baseline_only_splits"],
                "split": cfg["split"],
                "seeds": cfg["seeds"],
                "primary_metric": cfg["primary_metric"],
                "secondary_metrics": cfg["secondary_metrics"],
                "sweeps": cfg["sweeps"],
            },
            "results": {
                "raw_csv": "results/raw.csv",
                "sha256": sha256("results/raw.csv"),
                "rows": n_rows,
                "planned_runs": PLANNED_RUNS,
                "complete": n_rows == PLANNED_RUNS,
            },
            "analysis_scripts": ["src/analyze.py", "src/audit.py", "src/audit_errors.py", "src/claims_check.py"],
            "analysis_scripts_note": "Committed after results/raw.csv existed; run on the final raw.csv. Audits and claims_check are post-hoc.",
        }
        with open("FREEZE.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Wrote FREEZE.json")
        print(json.dumps({k: manifest[k] for k in ("design_commit", "code_commit",
                                                    "config_unchanged_since_design_commit")}, indent=2))
        print("dataset files match config:", manifest["dataset"]["train_csv"]["match"],
              manifest["dataset"]["unique_m_csv"]["match"])
        print("results complete:", manifest["results"]["complete"], f"({n_rows} rows)")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
