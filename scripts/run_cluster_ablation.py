"""Run a frozen, resumable raw-state clustering/selection ablation."""

import argparse
from hss.experiments.cluster_ablation import run_study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen_math_ablation.toml")
    parser.add_argument("--directory")
    args = parser.parse_args()
    run_study(args.config, args.directory)
