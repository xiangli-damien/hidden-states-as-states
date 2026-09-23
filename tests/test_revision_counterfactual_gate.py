from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
from revision_counterfactual_gate import cases,memory_split,parse,task_text,BUFFER,generate,capture,full_transform


def test_memory_pairs_split_by_sum_and_counterfactual_is_not_donor_answer():
    for u in range(10):
        split=[memory_split(a,(u-a)%10) for a in range(10)]
        assert [split.count(s) for s in ('train','validation','test')]==[6,2,2]
    items=cases();assert len(items)==80 and len({c['id'] for c in items})==80
    paired={}
    for c in items:
        paired.setdefault(c['recipient_id'],[]).append(c)
        r,d=c['recipient'],c['donor']
        assert memory_split(r['a'],r['b'])=='validation'
        assert memory_split(d['a'],d['b'])=='validation'
        assert r['tag']!=d['tag'] and r['c']!=d['c']
        if c['pair_type']=='different_u':
            assert c['cf_digit']!=(r['u']+r['c'])%10
            assert c['cf_digit']!=(d['u']+d['c'])%10
        else:
            assert r['u']==d['u']
        text=task_text(r,c['mode'])
        assert text.index(BUFFER)<text.index('Later input:')
    assert parse(' 7|blue ')==(7,'blue')
    assert parse('The answer is 7|blue')==(None,None)
    for pair in paired.values():
        assert len(pair)==2 and {p['pair_type'] for p in pair}=={'same_u','different_u'}
        assert pair[0]['recipient']==pair[1]['recipient']


def test_memory_region_precedes_offset_and_contains_the_tail():
    from revision_counterfactual_gate import encode
    from types import SimpleNamespace
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):return 'SYS\nUSER\n'+messages[1]['content']+'\nASSISTANT\n'
        def __call__(self,text,**kw):
            return SimpleNamespace(input_ids=list(range(len(text))),offset_mapping=[(i,i+1) for i in range(len(text))])
    tokenizer=Tokenizer()
    for case in cases():
        ids,tail,text=encode(tokenizer,case['recipient'],case['mode'])
        other,region,_=encode(tokenizer,case['recipient'],case['mode'],'memory_region')
        assert ids==other and region[-16:]==tail and len(region)>16
        assert region[-1]<text.index('Later input:')
        assert text.index('Remember') in region


def test_future_causality_and_final_block_past_output_null():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(3)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,eos_token_id=2,pad_token_id=0)).eval()
    ids=[3,4,5,6,7,8];future=[3,4,5,9,10,11]
    a,_=capture(model,ids,[0,1,2],layers=(1,2));b,_=capture(model,future,[0,1,2],layers=(1,2))
    for layer in (1,2):np.testing.assert_array_equal(a[layer],b[layer])
    class Tokenizer:
        def decode(self,ids,skip_special_tokens=True):return '0|red'
    tokenizer=Tokenizer()
    original=generate(model,tokenizer,ids)
    null=generate(model,tokenizer,ids,model.model.layers[1],[0,1,2],lambda h:torch.zeros_like(h))
    assert null['generated_ids']==original['generated_ids']
    changed=[3,12,5,6,7,8]
    with torch.inference_mode():
        replacement=model.model.embed_tokens(torch.tensor([12])).numpy()
    patched=generate(model,tokenizer,ids,model.model.embed_tokens,[1],full_transform(replacement))
    counterfactual=generate(model,tokenizer,changed)
    assert patched['generated_ids']==counterfactual['generated_ids']
