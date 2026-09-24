"""Compare paper NDR/CoE with final pre-RMS vs post-RMS response means on CPU."""
import argparse
import json
from pathlib import Path
from hss.analysis.channel_data import load_config
from hss.analysis.mean_geometry import run, render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mean_geometry.toml")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.report_only:
        root = Path(cfg['output_root'])
        render(root, json.loads((root/'analysis.json').read_text()))
    else:
        run(cfg)
