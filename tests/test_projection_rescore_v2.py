import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from projection_rescore_v2 import extract,match


@pytest.mark.parametrize('text,expected',[
    (r'Therefore the area of $PQRS$ is **17 square units**.','17'),
    (r'Therefore the polygon \(ABCDEFG\) has 14 diagonals.','14'),
    ('Therefore, Mr. Smith worked a total of 6 hours.','6'),
    ('So one pattern has 12 beads.\n\n### Final Answer\n\nShe needs a total of 734 beads for 1 bracelet and 9 necklaces.','734'),
    (r'Thus $f(9)=2$. '+ '\n\n### Conclusion\n\n'+r'The sum $f(1)+f(2)$ is $57$.','57'),
    (r'Conclusions about angles: $360^\circ$. '+ '\n\n### Conclusion\n\n'+'The polygon has 13 sides.','13'),
    (r'Thus the roots are $1,-1,\frac12,-\frac12$, which gives a total of **4** roots.','4'),
    ('Therefore, she should rent size 29 shoes assuming the relationship is directly proportional.','29'),
    (r'Therefore the expression $5x-|x-5|$ simplifies to $6x-5$ for $x<5$.','6x-5'),
    ('Therefore the polygon is a 15-sided polygon.','15'),
    ('Thus the possible sums are 621, 710, and 799.','621, 710, and 799.'),
    (r'The final answer is $443_{5}$.',r'443_{5}'),
    (r'The answer is:\n\[\boxed{286}\]','286'),
])
def test_terminal_claims(text,expected):
    assert extract(text,'math')['candidate']==expected


def test_exact_integer_and_base_format():
    assert match('1005','1004','math')==(False,[])
    assert match('2015','2013','math')==(False,[])
    assert match('1004.0','1004','math')==(True,[])
    assert match('0.3333333','1/3','math')==(True,[])
    assert match('443_{5}','443_5','math')==(True,[])
    assert extract('Thus possible values are 2, 3, and 4.','math')['flags']==['multiple_final_values']


def test_no_integer_tolerance_even_for_negative():
    assert match('-1005','-1004','math')==(False,[])
    assert extract(r'Earlier 30; finally \boxed{40}.','math')['candidate']=='40'


def test_sympy_nested_if_available():
    pytest.importorskip('sympy')
    assert match(r'\frac{\sqrt3}{3}',r'1/\sqrt{3}','math')==(True,[])


def test_complete_pipeline_and_reading_packet(tmp_path):
    pytest.importorskip('sympy')
    import json,hashlib
    from run_projection_scoring import run
    from audit_projection_rescore_v1 import run as audit
    root=tmp_path/'extension';primary=tmp_path/'primary';root.mkdir();primary.mkdir()
    manifest=[]
    for name,value,old in [('baseline','1004',False),('candidate','1005',True)]:
        p=primary/(name+'.json');p.write_text(json.dumps(dict(sample_id='synthetic',split='validation',condition=dict(name=name),
            prompt_text='Synthetic counting question',response_text='The final answer is '+value+'.',ground_truth='1004',
            correct=old,parsed_answer=value,finish_reason='eos')))
        manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (root/'records.json').write_text(json.dumps(manifest))
    (root/'summary.json').write_text(json.dumps(dict(comparisons=[dict(group='synthetic',method='candidate',control='baseline',
        per_question=[dict(method_record=manifest[1]['path'],control_record=manifest[0]['path'])]) ])))
    (root/'semantic_review_v2').mkdir();(root/'semantic_review_v2/summary.json').write_text(json.dumps(dict(rows=[])))
    out=root/'rescored';run(root,primary,out,2);audit(root,out)
    s=json.loads((out/'summary.json').read_text());packet=json.loads((out/'review_packet.json').read_text())
    assert s['changed_labels']==2 and s['new_full_answers_to_read']==2
    assert s['comparisons'][0]['rescored']['delta']==-1
    assert all(set(a)=={'answer_id','prompt','reference','response'} for a in packet['answers'])
    with pytest.raises(RuntimeError):run(root,primary,out,2)
    (out/'scores.jsonl').write_text('')
    with pytest.raises(AssertionError):audit(root,out)
