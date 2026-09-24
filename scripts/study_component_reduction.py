"""CPU-only follow-up: raw common-coordinate energy versus NDR."""
import argparse
from hss.analysis.channel_data import load_config
from hss.analysis.component_data import extract

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/component_reduction.toml')
    p.add_argument('--stage',required=True,choices=['extract','analyse','report'])
    p.add_argument('--limit-shards',type=int,default=0)
    a=p.parse_args();cfg=load_config(a.config)
    if a.stage=='extract': extract(cfg,a.limit_shards)
    else:
        from hss.analysis.component_reduction import analyse,render
        (analyse if a.stage=='analyse' else render)(cfg)
