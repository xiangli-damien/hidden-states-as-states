import numpy as np
from scripts.run_appendix_global_gmm import purity


def test_layer_purity_is_sample_weighted_and_has_no_empty_division():
    # Layers [0,0,0,0,1,1,1,1]; unequal component sizes, one empty component.
    result=purity(np.array([0,0,0,1,0,1,1,1]),2,4,3)
    assert result['sample_weighted_purity']==.75
    assert result['empty_clusters']==1
    assert result['layer_purity']==[.75,.75,0.]
    result=purity(np.array([0,0,0,0,0,0,1,1]),2,4,3)
    assert result['sample_weighted_purity']==.75
    assert np.isclose(result['layer_purity'][0],2/3)
    assert result['layer_purity'][1]==1
