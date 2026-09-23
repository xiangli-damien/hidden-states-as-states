import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_common import OncePatch, nearest, reconstruct, nmse_rows, paired_ratio_ci


def test_reconstruction_keeps_actual_tokens_distinct_and_uses_training_anchor():
    d={'centers': np.array([[0.,0.],[10.,0.]],np.float32),
       'train_mean': np.array([5.,0.],np.float32),
       'global_basis':np.array([[0.,1.]],np.float32),
       'local_basis':np.array([[[0.,1.]],[[0.,1.]]],np.float32)}
    x=np.array([[1.,3.],[11.,-2.]],np.float32)
    np.testing.assert_array_equal(nearest(x,d['centers']),[0,1])
    np.testing.assert_allclose(reconstruct(x,d,'local_pca_1'),[[0,3],[10,-2]])
    np.testing.assert_allclose(reconstruct(x,d,'global_pca_1'),[[5,3],[5,-2]])
    a,b=nmse_rows(x,reconstruct(x,d,'centroid'),d['train_mean'])
    np.testing.assert_allclose(a,[10,5]);np.testing.assert_allclose(b,[25,40])
    assert paired_ratio_ci(a,b,boot=20)['estimate']==pytest.approx(15/65)


def test_nmse_denominator_is_not_test_centered():
    x=np.array([[10.,0.],[10.,0.]])
    a,b=nmse_rows(x,np.zeros_like(x),np.zeros(2))
    assert paired_ratio_ci(a,b,boot=20)['estimate']==1
    assert paired_ratio_ci([0,0],[0,0],boot=20)['estimate'] is None


def test_patch_identity_position_and_decode_cache_contract():
    torch=pytest.importorskip('torch')
    h=torch.arange(30).reshape(1,5,6).float()
    patch=OncePatch([1,3],5,lambda z:z+2)
    out=patch(None,None,h)
    assert torch.equal(h[0,0],out[0,0])
    assert torch.equal(out[0,[1,3]],h[0,[1,3]]+2)
    assert patch.calls==1 and patch.energy==48
    assert patch(None,None,torch.zeros(1,1,6)) is None
    identity=OncePatch([0,1,2,3,4],5,lambda z:z)
    assert torch.equal(identity(None,None,(h,'cache'))[0],h)
    assert identity.energy==0
    with pytest.raises(ValueError,match='fresh'):
        OncePatch([0],5,lambda z:z)(None,None,torch.zeros(1,1,6))


def test_basis_rank_and_geometric_center_are_explicit():
    pytest.importorskip('sklearn')
    from fit_revision_geometry import basis
    x=np.array([[10.,0.],[11.,0.],[12.,0.]])
    b=basis(x,4,np.zeros(2))
    assert b.shape==(4,2)
    np.testing.assert_allclose(b[2:],0)
    np.testing.assert_allclose(((x@b.T)@b),x,atol=1e-5)
    assert not basis(np.zeros((0,2)),4,np.zeros(2)).any()
