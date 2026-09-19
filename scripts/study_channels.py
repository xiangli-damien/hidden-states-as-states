"""Reproducible CPU investigation; see configs/channel_study.toml."""
import argparse
from hss.analysis.channel_data import extract, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/channel_study.toml")
    p.add_argument("--stage", choices=["summary", "tokens", "analyse", "report"], required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("summary", "tokens"):
        extract(cfg, args.stage)
    elif args.stage == "analyse":
        from hss.analysis.channel_study import run
        run(cfg)
    else:
        from hss.analysis.channel_report import render
        render(cfg)


if __name__ == "__main__":
    main()
