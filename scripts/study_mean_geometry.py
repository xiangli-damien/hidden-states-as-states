"""Compare paper NDR/CoE with final pre-RMS vs post-RMS response means on CPU."""
import argparse
from hss.analysis.channel_data import load_config
from hss.analysis.mean_geometry import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mean_geometry.toml")
    args = parser.parse_args()
    run(load_config(args.config))

