"""CPU-only RMSNorm factorization and fixed-response counterfactual geometry."""
import argparse
from hss.analysis.channel_data import load_config
from hss.analysis.rms_mechanism import extract, analyse, render

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/rms_mechanism.toml')
    p.add_argument('--stage', required=True, choices=['extract','analyse','report'])
    p.add_argument('--limit-shards', type=int, default=0, help='Extraction smoke only; incomplete caches cannot be analysed')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.stage == 'extract': extract(cfg, args.limit_shards)
    elif args.stage == 'analyse': analyse(cfg)
    else: render(cfg)

