import argparse
import os
import shutil
import sys

from exception.exception import CustomException
from logger.logger import logger
from src.analyze import main as analyze
from src.data import load_config
from src.sweep import run_sweep


def main():
    ap = argparse.ArgumentParser(description="Stress test: sweep, then analysis.")
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, end levels only; writes to results/quick (about 5 minutes)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip the sweep; redo figures and tables from the existing raw.csv")
    ap.add_argument("--fresh", action="store_true",
                    help="delete results/raw.csv and results/preds first, to recompute everything")
    args = ap.parse_args()

    try:
        cfg = load_config()

        if args.fresh and not args.quick:
            logger.warning("--fresh: deleting results/raw.csv and results/preds "
                           "(committed copies can be restored with: git checkout -- results)")
            if os.path.exists("results/raw.csv"):
                os.remove("results/raw.csv")
            shutil.rmtree("results/preds", ignore_errors=True)

        if not args.analyze_only:
            logger.info("Step 1/2: sweep")
            run_sweep(cfg, quick=args.quick)

        logger.info("Step 2/2: analysis")
        analyze(quick=args.quick)
        logger.info("Done")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()