from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in [str(ROOT), str(SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--max-workers', type=int, default=1)
    args = parser.parse_args()
    results = run_batch([Path(p) for p in args.configs], max_workers=args.max_workers)
    for row in results:
        print(f"{row['name']} -> {row['output_dir']}")


if __name__ == '__main__':
    main()
