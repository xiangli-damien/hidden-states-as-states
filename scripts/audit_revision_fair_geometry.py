"""CPU verification of geometry coverage and real-parameter density contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
from hss.cluster.mfa import MFAModel
from revision_common import sha,write_json
from revision_factor_common import FairDecoders
from prepare_revision_fair_comparison import load_arrays
from measure_revision_fair_geometry import verify_inputs


def run(root):
    plan=verify_inputs(root);dest=root/'geometry'
    receipt=json.loads((dest/'_SUCCESS.json').read_text())
    assert receipt['complete'] and receipt['conditions']==plan['expected_conditions']
    for name,digest in receipt['files'].items():assert sha(dest/name)==digest,name
    rows=json.loads((dest/'per_question.json').read_text())
    index={(r['dataset'],r['split'],r['sample_id'],r['method']):r for r in rows}
    assert len(index)==len(rows)==receipt['conditions']
    for k,r in index.items():
        assert np.isfinite(r['ideal_mse']) and r['ideal_mse']>=0
        if r['method']=='identity':assert r['actual_bf16_mse']==r['ideal_mse']==0
        if r['method'].startswith('fa_orthogonal_'):
            rank=r['method'].rsplit('_',1)[1]
            posterior=index[(*k[:3],f'fa_{rank}')]
            # Orthogonal projection is the Euclidean minimizer in the very
            # same affine W span; posterior shrinkage cannot improve ideal MSE.
            assert r['ideal_mse']<=posterior['ideal_mse']+1e-10
    fits=json.loads((root/'fitting_summary.json').read_text())
    decoder=FairDecoders(load_arrays(root/'decoders.npz'))
    prior=float(np.sum(decoder.fixed[8].weights*np.log(decoder.fixed[8].weights)))
    training=json.loads((dest/'training_density.json').read_text())
    for r in training:
        source=fits[str(r['rank'])]
        np.testing.assert_allclose(r['joint_mixture_log_density'],source['training_joint_mean_log_density'],rtol=1e-10,atol=1e-6)
        assert r['tokens']==plan['training_tokens']
        assert r['fixed_mixture_log_density']>=source['training_assigned_component_mean_log_density']+prior-1e-6
    # Independent existing MFA implementation on actual high-dimensional heldout
    # states complements the dense small-covariance unit tests.
    x=load_arrays(root/'inputs/math/captures.npz')['x'][0].astype(float)
    max_error=0.
    for rank in [8,4,16]:
        for model in [decoder.fixed[rank],decoder.joint[rank]]:
            reference=MFAModel(model.weights,model.means,model.loadings,model.noise)
            p,score=model.posterior(x)
            expected=reference.score_samples(x)
            np.testing.assert_allclose(score,expected,rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(p,reference.predict_proba(x),rtol=1e-9,atol=1e-10)
            max_error=max(max_error,float(np.max(np.abs(score-expected))))
    outcome={'complete':True,'conditions':len(rows),'questions':receipt['questions'],
        'full_training_density_matches_original_joint_fit':True,
        'fa_orthogonal_projection_error_bound_verified':True,
        'real_parameter_reference_density_max_abs_error':max_error,
        'geometry_receipt_sha256':sha(dest/'_SUCCESS.json'),'code_sha256':sha(Path(__file__)),
        'scope':'CPU geometry/likelihood contracts; functional KL/NLL remains unmeasured.'}
    write_json(dest/'audit.json',outcome);print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):run(a.root)
