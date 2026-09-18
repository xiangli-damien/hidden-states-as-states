"""Check published scalar scores, fitted predictions, and report artifact integrity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from hss.experiments.artifacts import file_digest,save_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    results=json.loads((a.root/'analysis.json').read_text());checked={}
    for name,r in results.items():
        out=a.root/name;s=pd.read_parquet(out/'samples.parquet');pred=pd.read_parquet(out/'predictions.parquet')
        assert np.array_equal(s.sample_id,pred.sample_id) and np.array_equal(s.y,pred.y)
        assert not s.sample_id.duplicated().any()
        test=s.partition.eq('confirmation').to_numpy();train=~test;y=s.y.to_numpy()
        assert not set(s.loc[test,'prompt_sha256'])&set(s.loc[train,'prompt_sha256'])
        assert int(test.sum())==r['test_n'] and len(s)==r['n']
        for k,v in r['scalar_results'].items():
            use=test&np.isfinite(s[k].to_numpy())
            np.testing.assert_allclose(roc_auc_score(y[use],v['sign']*s.loc[use,k]),v['auc'],atol=1e-12)
            assert v['ci'][0]<=v['ci'][1] and int(use.sum())==v['test_n']
        for k,v in r['prediction_tests']['models'].items():
            np.testing.assert_allclose(roc_auc_score(y[test],pred.loc[test,k]),v['auc'],atol=1e-12)
        np.testing.assert_allclose(s.global_centered_energy,s.within_response_energy+s.mean_centered_energy,atol=1e-10)
        for k in ['q_raw_mean','q_token_mean','q_energy_ratio','q_rms_mean']:
            assert s[k].between(0,1).all()
        for stem in ['layers','positions','raw_energy','reduction']:
            for ext in ['png','svg']:assert (out/f'{stem}.{ext}').stat().st_size>1000
        checked[name]={'n':len(s),'test_n':int(test.sum()),'scalar_aucs':len(r['scalar_results']),
                       'fitted_aucs':len(r['prediction_tests']['models']),'figure_files':8}
    v={'passed':True,'datasets':checked,'analysis_sha256':file_digest(a.root/'analysis.json')}
    save_json(a.root/'validation.json',v);print(json.dumps(v,indent=2))
