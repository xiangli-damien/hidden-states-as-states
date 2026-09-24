from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_revision_locality_confirmation import select
from prepare_revision_locality_confirmation import evaluation_config
from revision_locality_common import conditions,key


def test_selection_is_order_invariant_and_excludes_prior_prompts_and_duplicate_questions():
    data=[{'sample_id':str(i),'question':f'Question {i}','prompt_text':f'Prompt {i}'} for i in range(10)]
    a,_=select(data,{'Prompt 3'},5,'fixed')
    b,_=select(list(reversed(data)),{'Prompt 3'},5,'fixed')
    assert a==b and all(r['sample_id']!='3' for r in a)
    duplicated=data+[{'sample_id':'dup','question':' Question   1 ','prompt_text':'new'}]
    c,excluded=select(duplicated,{'Prompt 3'},9,'fixed')
    assert len(c)==9 and set(excluded)=={'3','dup'}


def test_selection_never_uses_gold_or_model_outcome():
    data=[{'sample_id':str(i),'question':f'Q{i}','prompt_text':f'P{i}','ground_truth':i,'label':i%2} for i in range(20)]
    before,_=select(data,set(),8,'frozen')
    changed=[{**r,'ground_truth':-r['ground_truth'],'label':1-r['label']} for r in data]
    after,_=select(changed,set(),8,'frozen')
    assert [r['sample_id'] for r in before]==[r['sample_id'] for r in after]


def test_confirmation_keeps_source_rank_choice_and_all_control_seeds():
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/revision_locality_confirmation_20260923.json').read_text())
    ev=evaluation_config(cfg)
    assert ev['ranks']==[8] and ev['shared_ranks']==[8,64,512]
    primary=conditions(ev,True);aux=conditions(ev,False)
    assert len(primary)==40 and len(aux)==14
    names={key(c) for c in primary}
    assert all(f'wrong_local_8_{seed}' in names for seed in (42,137,271))
    assert all(f'remove_local_radial_random_8_{alpha}_{seed}' in names for alpha in (.25,.5,1.) for seed in (42,137,271))
    assert not any('rank' in c and c['rank'] not in [8,64,512] for c in primary)
