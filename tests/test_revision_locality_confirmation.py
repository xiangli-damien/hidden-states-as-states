from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_revision_locality_confirmation import select


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
