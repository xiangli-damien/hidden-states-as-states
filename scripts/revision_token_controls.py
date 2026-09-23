"""Train-only position/token-identity controls for raw token reconstruction.

Diagnostic baselines, not label models, not limited to the GMM parameter budget.
Unknown test token IDs fall back to the training grand mean. No test vector is
used to fit or select a decoder. Question-level error sums match the GMM metric.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fit_revision_geometry import read_view
from revision_common import freeze,provenance,write_json,write_npz,status,paired_ratio_ci,sha


SOURCE=Path('/lambda/nfs/dami/hss/revision-foundations-20260923')
DECODERS=Path('/lambda/nfs/dami/hss/revision-token-decoders-v2-20260923')
ROOT=Path('/lambda/nfs/dami/hss/revision-token-controls-20260923')
VIEWS=[(0,14,'tokens'),(0,28,'tokens'),(0,14,'question_tokens'),(0,28,'question_tokens'),(16,14,'tokens'),(16,28,'tokens')]


def fit_groups(x,groups):
    labels,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    sums=np.zeros((len(labels),x.shape[-1]),dtype=np.float64)
    np.add.at(sums,inverse,x)
    return labels,(sums/counts[:,None]).astype(np.float32),counts


def lookup(groups,labels,means,fallback):
    index=np.searchsorted(labels,groups)
    known=(index<len(labels))
    known &= labels[np.minimum(index,len(labels)-1)]==groups
    out=np.broadcast_to(fallback,(len(groups),len(fallback))).copy()
    out[known]=means[index[known]]
    return out,known


def token_matrix(frame,prefix,role):
    tokens={}
    for f in (SOURCE/'prefixes').glob('shard_*/tokens.json'):
        for row in json.loads(f.read_text()):tokens[row['sample_id']]=row
    result=[]
    for sid in frame.sample_id:
        row=tokens[sid]
        ids=row['prompt_ids']+row['response_ids'][:prefix]
        positions=row['question_positions'][-16:] if role=='question_tokens' else range(len(ids)-16,len(ids))
        result.append([ids[i] for i in positions])
    array=np.asarray(result,dtype=np.int64)
    assert array.shape==(len(frame),16)
    return array


def run_one(job):
    with threadpool_limits(limits=2):
        return _run_one(job)


def _run_one(job):
    prefix,layer,role=job;name=f'p{prefix}_l{layer}_{role}';dest=ROOT/name
    if (dest/'_SUCCESS.json').exists():return json.loads((dest/'summary.json').read_text())
    cfg=json.loads((SOURCE/'prefixes/plan.json').read_text())['config']
    frame,x=read_view(cfg,prefix,layer,role)
    train=frame.split.eq('train').to_numpy();test=frame.split.eq('test').to_numpy()
    ids=token_matrix(frame,prefix,role)
    flat=x[train].reshape(-1,x.shape[-1]);grand=flat.astype(np.float64).mean(0).astype(np.float32)
    position=x[train].astype(np.float64).mean(0).astype(np.float32)
    labels,means,counts=fit_groups(flat,ids[train].ravel())
    write_npz(dest/'decoder.npz',grand_mean=grand,position_means=position,token_labels=labels,token_means=means,token_counts=counts)
    values=x[test];test_ids=ids[test]
    identity,known=lookup(test_ids.ravel(),labels,means,grand)
    recovered={'position_mean':np.broadcast_to(position,values.shape),'token_identity_mean':identity.reshape(values.shape)}
    baseline=pd.read_parquet(DECODERS/'geometry'/name/'reconstruction_per_question.parquet')
    gmm_summary=json.loads((DECODERS/'geometry'/name/'summary.json').read_text())
    baseline=baseline.loc[baseline.split.eq('test') & baseline.method.eq('centroid')].set_index('sample_id').loc[frame.loc[test,'sample_id']]
    denominator=np.square(values.astype(np.float64)-grand).sum(axis=(1,2))
    # This metric is exactly comparable to the published v2 GMM token error.
    np.testing.assert_allclose(denominator,baseline.train_centered_energy.to_numpy(),rtol=1e-6)
    output=pd.DataFrame({'sample_id':frame.loc[test,'sample_id'].to_numpy(),'split':'test',
                         'train_centered_energy':denominator,'gmm_centroid_error':baseline.squared_error.to_numpy()})
    summaries=[]
    for method,z in recovered.items():
        error=np.square(values.astype(np.float64)-z).sum(axis=(1,2))
        output[method+'_error']=error
        summaries.append({'method':method,'nmse':paired_ratio_ci(error,denominator),
            'nmse_minus_GMM':paired_ratio_ci(error-baseline.squared_error.to_numpy(),denominator)})
    output['known_token_fraction']=known.reshape(test_ids.shape).mean(1)
    output.to_parquet(dest/'per_question.parquet',index=False)
    result={'view':name,'train_questions':int(train.sum()),'test_questions':int(test.sum()),
            'normalization':'none','test_distinct_token_windows':len(np.unique(test_ids,axis=0)),
            'train_token_vocabulary':len(labels),'test_known_token_fraction':float(known.mean()),
            'position_mean_parameters':int(position.size),'token_mean_parameters':int(means.size),
            'GMM_center_parameters':gmm_summary['selected']['k']*x.shape[-1],'GMM_K':gmm_summary['selected']['k'],
            'GMM_centroid_nmse':paired_ratio_ci(baseline.squared_error.to_numpy(),denominator),
            'methods':summaries,'scope':'Historical MATH test, exploratory; token-ID baseline is not capacity-matched and is not a causal intervention'}
    write_json(dest/'summary.json',result)
    write_json(dest/'_SUCCESS.json',{'summary_sha256':sha(dest/'summary.json'),'decoder_sha256':sha(dest/'decoder.npz'),
                                   'per_question_sha256':sha(dest/'per_question.parquet')})
    return result


def run():
    freeze(ROOT/'plan.json',provenance({'views':VIEWS,'source':str(SOURCE),'decoders':str(DECODERS),
           'methods':['position_mean','token_identity_mean'],'selection':'none; train means only; no correctness labels'},
           [Path(__file__),SOURCE/'prefixes/plan.json',DECODERS/'plan.json',Path(__file__).with_name('fit_revision_geometry.py')]))
    status(ROOT,'fit',state='running',completed=0,expected=len(VIEWS));results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_one,job) for job in VIEWS]):
            results.append(future.result());status(ROOT,'fit',state='running',completed=len(results),expected=len(VIEWS))
    write_json(ROOT/'summary.json',results);write_json(ROOT/'_SUCCESS.json',{'views':len(results)})
    status(ROOT,'fit',state='complete',completed=len(results),expected=len(VIEWS))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:run()
    except BaseException:
        status(ROOT,'fit',state='failed',traceback=traceback.format_exc());raise
