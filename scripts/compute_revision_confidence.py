"""Readout-derived confidence of the stored visible-prefix terminal state.

Uses the pinned real final RMSNorm and unembedding, with raw pre-norm vectors.
Saved values are computed in float32 after bf16 model readout; batching can
introduce bf16 rounding differences from the original singleton forward pass.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from extract_revision_prefixes import load_model
from revision_common import config, freeze, provenance, write_json, status


@torch.inference_mode()
def run(cfg):
    root=Path(cfg['output']);dest=root/'confidence';dest.mkdir(parents=True,exist_ok=True)
    freeze(dest/'plan.json',provenance(cfg,[Path(__file__),root/'prefixes/plan.json']))
    if (dest/'_SUCCESS.json').exists():
        return
    model,tokenizer=load_model(cfg)
    li=cfg['layers'].index(model.config.num_hidden_layers)
    for prefix in cfg['prefixes']:
        frames=[]
        for marker in sorted((root/'prefixes').glob('shard_*/_SUCCESS.json')):
            rows=pd.read_parquet(marker.parent/'rows.parquet')
            with np.load(marker.parent/f'prefix_{prefix}.npz') as a:
                valid=a['valid'];h=a['window'][valid,li,-1]
            rows=rows.loc[valid,['sample_id']].copy()
            values=[]
            for start in range(0,len(h),16):
                x=torch.from_numpy(h[start:start+16]).to(model.device,dtype=torch.bfloat16)
                logp=model.lm_head(model.model.norm(x)).float().log_softmax(-1)
                top=logp.topk(2,dim=-1).values
                block=torch.stack([-(logp.exp()*logp).sum(-1),top[:,0]-top[:,1],top[:,0].exp(),x.float().square().mean(-1).sqrt()],dim=1)
                values.append(block.cpu().numpy())
            if len(h):
                values=np.concatenate(values)
                for i,name in enumerate(['next_token_entropy','next_token_logit_margin','next_token_max_probability','terminal_rms']):
                    rows[name]=values[:,i]
                frames.append(rows)
        pd.concat(frames,ignore_index=True).to_parquet(dest/f'prefix_{prefix}.parquet',index=False)
    write_json(dest/'_SUCCESS.json',{'completed_unix':time.time(),'prefixes':cfg['prefixes'],
                                   'note':'Pinned real RMSNorm + lm_head; no fitted confidence direction proxy.'})
    status(root,'confidence',state='complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    run(config(p.parse_args().config))
