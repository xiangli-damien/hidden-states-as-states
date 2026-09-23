"""Versioned full-window token fits; reuses immutable prefix extraction only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import traceback
from fit_revision_geometry import fit_one
from revision_common import config,freeze,provenance,status,write_json


def run(cfg):
    root=Path(cfg['output']);prefix=Path(cfg['prefix_root'])
    if not (prefix/'_SUCCESS.json').exists():
        raise ValueError('Verified complete extraction required')
    freeze(root/'plan.json',provenance(cfg,[Path(__file__),Path(__file__).with_name('fit_revision_geometry.py'),
        Path(__file__).with_name('revision_common.py'),prefix/'plan.json',prefix/'_SUCCESS.json']))
    jobs=[(cfg,*v) for v in cfg['views']]
    status(root,'fit',state='running',completed=0,expected=len(jobs))
    summaries=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for future in as_completed([pool.submit(fit_one,j) for j in jobs]):
            result=future.result();summaries.append(result)
            if result['tokens_per_question_fit']!=16:
                raise AssertionError('Every token position must contribute to training')
            status(root,'fit',state='running',completed=len(summaries),expected=len(jobs),last=result['name'])
    write_json(root/'summary.json',summaries)
    write_json(root/'_SUCCESS.json',{'views':len(summaries),'positions_per_question':16})
    status(root,'fit',state='complete',completed=len(summaries),expected=len(jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    cfg=config(p.parse_args().config)
    try:
        run(cfg)
    except BaseException:
        status(cfg['output'],'fit',state='failed',traceback=traceback.format_exc());raise
