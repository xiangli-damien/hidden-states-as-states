import importlib.util
from pathlib import Path
import pytest
from types import SimpleNamespace
import json

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


def study_with_rank_dependent_optima():
    study=mod.Study.__new__(mod.Study)
    study.args=SimpleNamespace(optimizer='squarem')
    study.units=[('post',0),('post',1)]
    study.records={}
    # Rank 16 looks worst at historical K=21, but wins at its own optimum.
    for unit in study.units:
        for rank,best_k,score in [(4,8,100),(8,16,50),(16,32,0)]:
            for k,icl in [(21,1000+rank),(best_k,score)]:
                row=dict(view=unit[0],layer=unit[1],rank=rank,k=k,icl=icl,
                         status='complete',seed=42,converged=True,tol=1e-4,initializations_completed=1)
                study.records[str(len(study.records))]=row
    return study


def test_every_layer_rank_searches_k_without_gmm_shortlisting():
    study=study_with_rank_dependent_optima()
    tasks=study.screen_tasks()
    for unit in study.units:
        for rank in mod.RANKS:
            planned={t['k'] for t in tasks if (t['view'],t['layer'])==unit and t['rank']==rank}
            cached={r['k'] for r in study.rows(unit) if r['rank']==rank}
            assert set(mod.COARSE_K)<=planned|cached
    refinements=study.refinement_tasks()
    for rank,expected in [(4,{7,9}),(8,{13,19}),(16,{26,38})]:
        assert {t['k'] for t in refinements if t['rank']==rank}==expected


def test_strict_finalists_cover_each_rank_before_runner_up():
    study=study_with_rank_dependent_optima()
    tasks=study.finalist_tasks()
    first=tasks[:len(study.units)*len(mod.RANKS)]
    assert {(t['layer'],t['rank']) for t in first}=={(u[1],r) for u in study.units for r in mod.RANKS}
    assert all(t['k']=={4:8,8:16,16:32}[t['rank']] for t in first)
    assert all(t['n_init']==3 and t['tol']==1e-5 for t in tasks)
    assert all(t['k']==21 for t in tasks[len(first):])


def test_search_revision_archives_protocol_but_rejects_fit_or_deadline_changes(tmp_path):
    old=dict(deadline_utc='fixed',ranks=[4,8,16],optimizer_sha256='unchanged',
             screen_ranks_per_layer=2,driver_sha256='old')
    mod.record_protocol(tmp_path,old)
    new=dict(old,screen_ranks_per_layer=3,driver_sha256='new',search={'rank_pruning':False})
    with pytest.raises(ValueError):mod.record_protocol(tmp_path,new)
    mod.record_protocol(tmp_path,new,revise_search=True)
    assert json.loads((tmp_path/'protocol_history'/f'{mod.digest(old)}.json').read_text())==old
    for changed in [dict(new,deadline_utc='later'),dict(new,optimizer_sha256='different')]:
        with pytest.raises(ValueError):mod.record_protocol(tmp_path,changed,revise_search=True)
    assert json.loads((tmp_path/'protocol.json').read_text())==new
