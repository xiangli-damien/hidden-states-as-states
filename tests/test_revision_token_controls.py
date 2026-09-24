from pathlib import Path
import sys

import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from revision_token_controls import fit_groups,lookup


def test_only_training_tokens_fit_and_unseen_tokens_fall_back():
    x=np.array([[2.,0.],[4.,2.],[9.,3.]],dtype=np.float32)
    ids,means,counts=fit_groups(x,np.array([4,4,9]))
    np.testing.assert_array_equal(ids,[4,9])
    np.testing.assert_allclose(means,[[3,1],[9,3]])
    np.testing.assert_array_equal(counts,[2,1])
    result,known=lookup(np.array([0,4,8,9,50]),ids,means,np.array([-1.,-2.],np.float32))
    np.testing.assert_allclose(result,[[-1,-2],[3,1],[-1,-2],[9,3],[-1,-2]])
    np.testing.assert_array_equal(known,[False,True,False,True,False])
