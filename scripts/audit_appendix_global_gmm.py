"""Independent CPU checks of saved pooled-GMM selection and layer purity."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from hss.cluster.gmm import GMMModel
from scripts.revision_common import sha, write_json


def audit(root):
    root=Path(root)
    read=lambda p:json.loads(p.read_text())
    completion=read(root/'COMPLETE.json')
    assert completion['summary_sha256']==sha(root/'summary.json')
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    assert summary['protocol_sha256']==sha(root/'protocol.json')
    for path,checksum in protocol['sources'].items():
        assert sha(path)==checksum, path
    rows=read(root/'candidates.json')
    assert [row['k'] for row in rows]==list(range(2,81))
    failures=[];unconverged_starts=[]
    for row in rows:
        p=Path(row['path']);marker=read(p/'complete.json')
        assert marker['fit_sha256']==sha(p/'fit.json')
        assert read(p/'fit.json')==row and len(row['restarts'])==3
        for i,restart in enumerate(row['restarts']):
            if not restart['converged']:unconverged_starts.append([row['k'],i])
        if not row['converged']:
            failures.append(row['k']);continue
        assert marker['model_sha256']==sha(p/'model.npz')
        params=row['k']*(2*3584)+row['k']-1
        assert row['n_parameters']==params and row['n']==145000
        bic=-2*row['log_likelihood']+params*np.log(145000)
        np.testing.assert_allclose(bic,row['bic'],rtol=1e-12,atol=1e-5)
        np.testing.assert_allclose(bic+2*row['entropy'],row['icl'],rtol=1e-12,atol=1e-5)
    valid=[r for r in rows if r['converged'] and np.isfinite(r['icl'])]
    best=min(r['icl']for r in valid)
    checks=[]
    for item in summary['results']:
        tolerance=item['tolerance'];selected=min((r for r in valid if r['icl']<=best+tolerance*max(abs(best),1.)),key=lambda r:r['k'])
        assert item['selected_path']==selected['path'] and item['k']==selected['k']
        arrays=dict(np.load(Path(selected['path'])/'model.npz'))
        model=GMMModel.from_state(selected['model'],arrays)
        assert np.isfinite(model.means_).all() and np.isfinite(model.covariances_).all()
        assert (model.covariances_>0).all() and (model.weights_>0).all()
        np.testing.assert_allclose(model.weights_.sum(),1.)
        assignments=np.load(root/f'assignments_t{tolerance}.npz')
        for method in ['nearest','posterior']:
            labels=assignments[method];assert labels.shape==(145000,)
            assert labels.min()>=0 and labels.max()<item['k']
            counts=np.bincount(labels*29+np.repeat(np.arange(29),5000),minlength=item['k']*29).reshape(item['k'],29)
            assert (counts.sum(0)==5000).all()
            np.testing.assert_array_equal(counts,item[method]['cluster_layer_counts'])
            weighted=sum(max(row) for row in counts)/145000
            np.testing.assert_allclose(weighted,item[method]['sample_weighted_purity'],atol=1e-15)
        # Held-out rows across all 29 layers; independent NumPy density/distance.
        indices=np.array([0,17,104,787,1234,2345,3567,4001,4765,4999])
        xs=np.concatenate([np.load(Path(protocol['cache'])/f'layer_{l}.npy',mmap_mode='r')[indices].astype(float)for l in range(29)])
        all_indices=np.concatenate([indices+l*5000 for l in range(29)])
        p=model.predict_proba(xs)
        distance=((xs[:,None,:]-model.means_[None,:,:])**2).sum(2)
        np.testing.assert_array_equal(p.argmax(1),assignments['posterior'][all_indices])
        np.testing.assert_array_equal(distance.argmin(1),assignments['nearest'][all_indices])
        checks.append(dict(tolerance=tolerance,k=item['k'],verified_assignment_rows=290,
                           nearest_purity=item['nearest']['sample_weighted_purity'],
                           posterior_purity=item['posterior']['sample_weighted_purity']))
    report=dict(passed=True,scope='All 79 fit receipts/model hashes/ICL scores, full counts/purity; independent assignments on 290 rows across all layers',
                failed_candidates=failures,nonconverged_starts=unconverged_starts,results=checks,
                best_at_upper_bound=summary['best_at_upper_bound'],summary_sha256=sha(root/'summary.json'))
    write_json(root/'independent_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    with threadpool_limits(limits=2):
        print(json.dumps(audit(args.root),indent=2))
