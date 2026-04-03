from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in [str(ROOT), str(SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.config import ExperimentConfig, load_config
from experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--stages', nargs='*', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if args.run_path:
        cfg = replace(cfg, data=replace(cfg.data, run_path=args.run_path))
    if args.output_dir:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.stages:
        cfg = replace(cfg, execution=replace(cfg.execution, stages=list(args.stages)))
    run_experiment(cfg)


if __name__ == '__main__':
    main()
