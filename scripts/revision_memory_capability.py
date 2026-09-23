"""Training-only diagnosis after a failed arithmetic capability prerequisite.

No activations are fitted or patched. Copy, elementary modular addition and two
fixed composed-task wordings isolate what failed; no validation/test execution.
"""
import hashlib
from pathlib import Path
import json
import traceback
from extract_revision_prefixes import load_model
from revision_common import provenance,freeze,write_json,status,sha
from revision_counterfactual_gate import memory_split,generate,BUFFER

ROOT=Path('/lambda/nfs/dami/hss/revision-memory-capability-20260923')


def tasks():
    result=[]
    def add(group,text,gold,tag,metadata):
        result.append({'id':hashlib.sha256((group+'/'+text+'/'+json.dumps(metadata,sort_keys=True)).encode()).hexdigest()[:20],
            'group':group,'text':text,'gold_digit':gold,'gold_tag':tag,**metadata})
    for u in range(10):
        for tag in ['red','blue']:
            text=f'Remember the digit {u}.'+BUFFER+f' Final instruction: copy the remembered digit followed by |{tag}. Output only that digit and tag.'
            add('delayed_copy',text,u,tag,{'u':u})
        for c in range(10):
            tag='red' if (u+c)%2 else 'blue'
            text=f'Compute ({u} + {c}) modulo 10. Output only the resulting single digit followed by |{tag}.'
            add('elementary_modulo_addition',text,(u+c)%10,tag,{'u':u,'c':c})
    pairs=[(a,b) for a in range(10) for b in range(10) if memory_split(a,b)=='train']
    for mode in ['explicit_digit','implicit_sum']:
        for template in ['formula','numbered_steps']:
            for i,(a,b) in enumerate(pairs):
                u=(a+b)%10;c=(3*a+7*b+1)%10;tag='red' if i%2 else 'blue'
                if template=='formula':
                    first=f'Let u = {u}.' if mode=='explicit_digit' else f'Let u = ({a} + {b}) modulo 10.'
                    last=(f' Now let c = {c}, and let tag = {tag}. Compute y = (u + c) modulo 10. '
                          'Output y|tag, replacing y and tag with their values. Do not output u. Give no explanation.')
                else:
                    first=(f'Step 1: the intermediate digit is {u}.' if mode=='explicit_digit' else
                           f'Step 1: add {a} and {b}, then keep the units digit as the intermediate digit.')
                    last=(f' Step 2: add {c} to the intermediate digit and keep the units digit of the new sum. '
                          f'Step 3: write the Step 2 result followed by |{tag}. Return only this final output.')
                add(mode+'/'+template,first+BUFFER+last,(u+c)%10,tag,{'a':a,'b':b,'u':u,'c':c,'split':'train'})
    assert len(result)==360 and len({r['id'] for r in result})==len(result)
    return result


def run():
    ROOT.mkdir(exist_ok=True)
    cfg={'model':'Qwen/Qwen2-7B-Instruct','revision':'f2826a00ceef68f0f2b946d945ecc0477ce4450c',
         'do_sample':False,'repetition_penalty':1.0,'max_new_tokens':16,'tasks':tasks(),
         'scope':'Capability diagnosis only. Composed prompts use train pairs only; no held-out claims, no intervention or codebook fit.',
         'search_budget':'Exactly two predeclared composed-task wordings; no adaptive expansion.'}
    freeze(ROOT/'plan.json',provenance(cfg,[Path(__file__),Path(__file__).with_name('revision_counterfactual_gate.py'),
                                           Path(__file__).with_name('extract_revision_prefixes.py')]))
    model,tokenizer=load_model(cfg);records=[]
    for task in cfg['tasks']:
        path=ROOT/'samples'/(task['id']+'.json')
        if path.exists():records.append(json.loads(path.read_text()));continue
        ids=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':task['text']}],tokenize=True,add_generation_prompt=True)
        out=generate(model,tokenizer,ids,repetition_penalty=1.0)
        record={'task':task,'input_ids':ids,'output':out,
                'correct':out['digit']==task['gold_digit'] and out['tag']==task['gold_tag']}
        write_json(path,record);records.append(record)
        if len(records)%20==0:
            status(ROOT,'capability',state='running',completed=len(records),expected=len(cfg['tasks']))
    assert len(records)==360 and {r['task']['id'] for r in records}=={r['id'] for r in cfg['tasks']}
    summary=[]
    for group in sorted({r['task']['group'] for r in records}):
        subset=[r for r in records if r['task']['group']==group]
        unique={r['task']['text']:r for r in subset}
        for r in subset:assert r['output']['generated_ids']==unique[r['task']['text']]['output']['generated_ids']
        correct=sum(r['correct'] for r in unique.values())
        summary.append({'group':group,'scheduled_cases':len(subset),'unique_prompts':len(unique),
            'correct':correct,'accuracy':correct/len(unique),
            'parse_failures':sum(r['output']['parse_failed'] for r in unique.values()),
            'meets_training_capability_threshold':correct/len(unique)>=.9})
    write_json(ROOT/'summary.json',summary)
    write_json(ROOT/'_SUCCESS.json',{'conditions':len(records),'summary_sha256':sha(ROOT/'summary.json'),
        'interpretation':'Training diagnostic complete, not an independent validation pass or a steering result'})
    status(ROOT,'capability',state='complete',completed=len(records),expected=360)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:run()
    except BaseException:
        status(ROOT,'capability',state='failed',traceback=traceback.format_exc());raise
