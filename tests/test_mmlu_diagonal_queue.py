import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mmlu_diagonal_gmm import select,refinement,needs_extension


def row(k,score,converged=True):return dict(k=k,icl=score,converged=converged)


def test_icl_negative_scores_and_convergence():
    rows=[row(4,-99),row(8,-100),row(16,-200,False)]
    assert select(rows,0)['k']==8
    assert select(rows,.02)['k']==4


def test_adaptive_search_densifies_and_extends():
    cfg=dict(refine_top_icl=1,icl_tolerances=[0,.02],initial_k=[16,32,64,80,160],boundary_margin=16)
    rows=[row(16,200),row(32,180),row(64,100),row(80,102),row(160,500)]
    wanted=refinement(rows,[r['k'] for r in rows],cfg)
    assert 63 in wanted and 65 in wanted and 64 not in wanted
    assert not needs_extension(rows,cfg)
    assert needs_extension([row(144,100),row(160,101)],cfg)
