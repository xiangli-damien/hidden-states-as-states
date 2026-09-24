"""Synthetic extraction regression cases, not an independent score validation set."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from projection_rescore_v1 import extract, canonical, tex_expression, match


@pytest.mark.parametrize('text,task,want', [
    (r'The answer is:\n\[\boxed{286}\]', 'gsm8k', '286'),
    ('Thus, she needs 30 minutes to reach her 20 mile goal.', 'gsm8k', '30'),
    ('Therefore, she needs 742 beads to make 10 necklaces.', 'gsm8k', '742'),
    (r'Therefore, the value is $\frac{\sqrt{3}}{3}$.', 'math', r'\frac{\sqrt{3}}{3}'),
    (r'Thus, $5x-|x-5| = 6x-5$ for $x<5$.', 'math', '6x-5'),
    (r'The answer is $\begin{pmatrix}2\\3\\1\end{pmatrix}$.', 'math', r'\begin{pmatrix}2\\3\\1\end{pmatrix}'),
    (r'The answer is 1. But the final answer is $\frac{2}{5}$.', 'math', r'\frac{2}{5}'),
    (r'\boxed{\frac{1}{1+\frac{1}{2}}}', 'math', r'\frac{1}{1+\frac{1}{2}}'),
    ('Thus, the list contains 4 values.', 'math', '4'),
    (r'Answer: $443_5$.', 'math', '443_5'),
    ('Therefore, the answer is -0.25.', 'gsm8k', '-0.25'),
    ('The final answer is 8, not 9.', 'gsm8k', '8'),
])
def test_extraction(text, task, want):
    assert extract(text, task)['candidate'] == want


def test_assignment_and_base_preserved():
    assert canonical('x=50', 'math') == '50'
    assert canonical('a=1,b=2', 'math') == 'a=1,b=2'
    assert canonical('443_5', 'math') == '443_5'
    assert canonical('30%', 'gsm8k') == '30'
    assert canonical('1,234', 'math') == '1234'


def test_no_reference_or_label_input_and_explicit_wrong_wins():
    assert extract(r'We tried 50, but the answer is \boxed{40}.', 'math')['candidate'] == '40'
    assert extract('No final answer.', 'math', 'legacy')['rule'] == 'frozen_parser_fallback'
    assert extract('Therefore the answer is 30 for the 20 mile goal.', 'gsm8k')['flags'] == ['multiple_numbers_in_conclusion']


def test_nested_symbolic_and_unsafe_input():
    assert tex_expression(r'\frac{1}{1+\frac{1}{2}}') == '((1)/(1+((1)/(2))))'
    for value in ["__import__('os')", 'x.__class__', 'hello(2)', r'443_5']:
        with pytest.raises(ValueError): tex_expression(value)


def test_numeric_no_sympy_required():
    assert match('x=50', '50', 'math') == (True, [])
    assert match('50', '40', 'math') == (False, [])
    assert match('1/2', '0.5', 'math') == (True, [])


def test_symbolic_if_available():
    pytest.importorskip('sympy')
    assert match(r'\frac{1}{1+\frac{1}{2}}', '2/3', 'math') == (True, [])
    expr = r'\sqrt{\frac{1+\frac{\sqrt{184}}{25}}{2}}-\sqrt{\frac{1-\frac{\sqrt{184}}{25}}{2}}'
    assert match(expr, '2/5', 'math') == (True, [])
    assert match('x+1', 'x+2', 'math') == (False, [])


def test_complete_manifest_pipeline_and_tamper(tmp_path):
    pytest.importorskip('sympy')
    import json, hashlib
    from run_projection_rescore_v1 import run
    from audit_projection_rescore_v1 import run as audit
    root=tmp_path/'fixture';root.mkdir();primary=tmp_path/'primary';primary.mkdir()
    records=[]
    for name, text, old in [('baseline','Therefore, 30 minutes for the 20 mile goal.',False),
                             ('candidate',r'Wrong guess 30; final answer is \boxed{40}.',True)]:
        p=primary/(name+'.json')
        raw=dict(sample_id='synthetic',split='validation',condition=dict(name=name),
                 response_text=text,prompt_text='Synthetic question',ground_truth='30',
                 correct=old,parsed_answer='20' if name=='baseline' else '30',finish_reason='eos')
        p.write_text(json.dumps(raw));records.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (root/'records.json').write_text(json.dumps(records))
    (root/'summary.json').write_text(json.dumps(dict(comparisons=[dict(group='synthetic',method='candidate',control='baseline',
        per_question=[dict(method_record=records[1]['path'],control_record=records[0]['path'])])])))
    out=root/'rescored';run(root,primary,out,2);audit(root,out)
    s=json.loads((out/'summary.json').read_text())
    assert s['records']==2 and s['false_to_true']==1 and s['true_to_false']==1
    assert s['comparisons'][0]['rescored']['delta']==-1
    assert not s['full_semantic_rescoring_complete']
    with pytest.raises(RuntimeError):run(root,primary,out,2)
    # A changed summary is rejected even if coverage and original inputs remain intact.
    s['comparisons'][0]['rescored']['delta']=1
    (out/'summary.json').write_text(json.dumps(s))
    with pytest.raises(AssertionError):audit(root,out)
