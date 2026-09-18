"""CPU-only confidence/direction follow-up; never loads a decoder on the GPU."""
import argparse
from hss.analysis.channel_data import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/confidence_study.toml')
    p.add_argument('--stage', required=True, choices=['scalars','analyse','layers','layer-analysis','report'])
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.stage in ['scalars', 'layers']:
        from hss.analysis import confidence_data
        getattr(confidence_data, args.stage)(cfg)
    elif args.stage == 'report':
        from hss.analysis.confidence_report import render
        render(cfg)
    else:
        from hss.analysis import confidence_study
        getattr(confidence_study, args.stage.replace('-', '_'))(cfg)


if __name__ == '__main__':
    main()
