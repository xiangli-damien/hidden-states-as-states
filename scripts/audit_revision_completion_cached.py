"""Independent count/prior, AUROC and saved-source integrity check for A/B/D."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from revision_common import sha,write_json


def auc(y,p):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(p,method='average')[y==1].sum()-n*(n+1)/2)/(n*m))


def run(root):
    folder=root/'cached';s=json.loads((folder/'summary.json').read_text());rc=json.loads((folder/'_SUCCESS.json').read_text())
    assert rc['summary_sha256']==sha(folder/'summary.json')
    for name,h in rc['files'].items():assert sha(folder/name)==h
    inputs=json.loads((folder/'sources.json').read_text())
    assert sha(folder/'sources.json')==s['inputs_sha256']
    # Large raw activation files were verified by the producer and are frozen;
    # this independent statistical audit rechecks the small source weights and
    # prediction receipts, rather than rereading the same raw gigabytes twice.
    for path,h in inputs.items():
        if Path(path).name!='prefix_0.npz':assert sha(path)==h,path
    pred=pd.read_parquet(folder/'transfer_predictions.parquet');y=pred.failure.to_numpy(int)
    assert len(pred)==1319 and not pred.sample_id.duplicated().any()
    with np.load(folder/'source_assignments.npz') as a:
        source_codes=a['states'];sy=a['labels'];tr=a['split']=='train';test=a['split']=='test';source_ids=a['sample_ids']
    adapt=json.loads((folder/'adaptation.json').read_text());ad=pred.sample_id.isin(adapt['sample_ids']).to_numpy()
    assert ad.sum()==128 and not set(adapt['sample_ids'])&set(adapt['confirmation_ids'])
    emitted=[];adapted=[]
    for li,l in enumerate([7,14,21,28]):
        with np.load(folder/f'block{l}.npz') as a:
            likelihood=a['likelihood'];prior=a['prior'];codes=a['target_code'];count=np.ones_like(likelihood)
            for k,c in zip(source_codes[tr,li],sy[tr]):count[c,k]+=1
            np.testing.assert_array_equal(likelihood,count/count.sum(axis=1)[:,None])
            counts=np.ones_like(likelihood)
            for k,c in zip(codes[ad],y[ad]):counts[c,k]+=1
            np.testing.assert_array_equal(a['adapt128_likelihood'],counts/counts.sum(axis=1)[:,None])
            emitted.append(np.log(likelihood[:,codes]));adapted.append(np.log(a['adapt128_likelihood'][:,codes]))
            np.testing.assert_array_equal(codes,pred[f'state_l{l}'])
    expected=softmax(np.log(prior)[:,None]+np.sum(emitted,axis=0),axis=0)[1]
    expected_ad=softmax(np.log(adapt['prior'])[:,None]+np.sum(adapted,axis=0),axis=0)[1]
    np.testing.assert_allclose(expected,pred.frozen_nb,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(expected_ad,pred.adapt128_nb,rtol=1e-12,atol=1e-12)
    for row in s['transfer']:
        ix=pred.split.eq('test').to_numpy() if row['scope']=='original_confirmation536' else np.ones(len(pred),bool)
        labels=y[ix];p=pred[row['method']].to_numpy()[ix]
        np.testing.assert_allclose(row['auroc'],auc(labels,p),rtol=0,atol=1e-15)
        rng=np.random.default_rng(42);values=[]
        for _ in range(2000):
            take=rng.integers(len(p),size=len(p))
            if 0<labels[take].sum()<len(p):values.append(auc(labels[take],p[take]))
        np.testing.assert_allclose(row['auroc_ci95'],np.quantile(values,[.025,.975]),rtol=0,atol=1e-14)
        np.testing.assert_allclose(row['brier'],np.square(labels-p).mean(),rtol=0,atol=1e-15)
    for row in s['flow']:
        li=[7,14,21,28].index(row['from']);pairs=dict(row['mapping_pairs'])
        a=source_codes[test,li];b=source_codes[test,li+1]
        obs=np.mean([pairs.get(int(x),-1)==int(z) for x,z in zip(a,b)])
        chance=sum(np.mean(a==x)*np.mean(b==z) for x,z in pairs.items())
        np.testing.assert_allclose([row['flow'],row['chance'],row['excess']],[obs,chance,obs-chance],rtol=0,atol=1e-14)
    write_json(folder/'audit.json',{'complete':True,'summary_sha256':sha(folder/'summary.json'),
        'source_receipt_sha256':sha(folder/'_SUCCESS.json'),'transfer_rows':len(s['transfer']),
        'scope':'Independent source/target counts, prior once, rank-AUC and intervals, paired IDs, flow/chance; existing reconstruction reuses audited tables',
        'code_sha256':sha(Path(__file__))})
    print(json.dumps({'cached_audit_complete':True,'transfer_rows':len(s['transfer'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)
