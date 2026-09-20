import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('deadline_driver',Path(__file__).parents[1]/'scripts/run_mfa_deadline.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_screening_and_incomplete_restarts_cannot_enter_main_selection():
    base=dict(status='complete',converged=True,tol=1e-5,initializations_completed=3,k=10,rank=8,icl=100,bic=90)
    provisional=dict(base,tol=1e-4,icl=10)
    unfinished=dict(base,initializations_completed=2,icl=5)
    assert mod.select([base,provisional,unfinished])==base
    assert mod.select([provisional,unfinished]) is None
    assert mod.select([base,provisional],strict=False)==provisional


def test_rank_map_roundtrip_and_validation():
    from hss.experiments.config import Experiment,ClusterConfig
    x=Experiment();x.data.paths=['/tmp/data'];x.cluster=ClusterConfig(method='mfa',rank_by_layer={'0':4,'1':8})
    assert Experiment.from_dict(x.to_dict()).cluster.rank_by_layer=={'0':4,'1':8}
    x.cluster.rank_by_layer={'0':-1}
    with pytest.raises(ValueError):Experiment.from_dict(x.to_dict())
