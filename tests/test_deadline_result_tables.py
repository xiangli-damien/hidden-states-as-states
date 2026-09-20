from types import SimpleNamespace
import numpy as np
from hss.analysis.tables import dynamics, selection_surface


def test_missing_derived_profile_values_are_rebuilt_from_saved_states():
    result=SimpleNamespace(states=np.array([[0,0],[1,2],[0,2]]),
        summary={'profile':[{'layer':0,'k':2},{'layer':1,'k':2}]},
        json=lambda _: {'local_to_global':[[0,1],[0,2]]})
    _,profile=dynamics(result)
    assert np.isnan(profile.self_transition.iloc[0])
    assert profile.self_transition.iloc[1]==1/3
    assert profile.relative_depth.tolist()==[0,1]


def test_rank_and_strict_admission_are_preserved_in_selection_surface():
    a=dict(k=2,rank=4,icl=100,converged=True,initializations_completed=3,tol=1e-5,fit_path='a')
    b=dict(a,rank=8,icl=90,fit_path='b')
    screening=dict(a,icl=1,tol=1e-4,initializations_completed=1,fit_path='screen')
    result=SimpleNamespace(config={'cluster':dict(selection_criterion='icl',require_convergence=True,
        tol=1e-5,n_init=3,parsimony_tolerance=.02)},
        json=lambda _: [dict(layer=0,selected=b,candidates=[a,b,screening])])
    table=selection_surface(result)
    assert table.selected.tolist()==[False,True,False]
    assert table.eligible.tolist()==[True,True,False]
    assert table.relative_criterion.iloc[1]==0
    assert not table.near_optimal.iloc[2]
