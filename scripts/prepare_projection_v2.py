"""Freeze v2 inputs on CPU while the protected experiment keeps running."""
import argparse, hashlib, inspect, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from openact_core.tasks.templates import get_template
from revision_common import freeze, sha, provenance, write_json
from revision_completion_common import ordered
from projection_v2_common import catalog, transfer_conditions, derangement


def normalized(text):
    return ' '.join(text.split())


def text_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def run(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise RuntimeError('Already frozen; inspect existing plan')
    main=Path(cfg['protected_primary']);parent=json.loads((main/'plan.json').read_text());pc=parent['config']
    selection=json.loads((main/'selection.json').read_text())
    assert selection['selected']==dict(name='c1_1.0',family='c1',alpha=1.) and selection['test_questions']==256
    assert selection['frozen_before_any_test_generation'] and selection['validation_audit_sha256']==sha(main/'validation_audit.json')
    for path,digest in parent['files'].items():assert sha(path)==digest,path
    assert (pc['layer'],pc['prefix'],pc['width'],pc['rank'])==(14,16,4,8)
    files=[main/name for name in ['plan.json','selection.json','inputs.json','support.json','generation_config.json',
                                  'execution_plan.json','smoke_audit.json','validation_audit.json','validation_SUCCESS.json']]
    files+=list(map(Path,parent['files']))
    files+=list(map(Path,json.loads((main/'execution_plan.json').read_text())['files']))
    decoder=Path(pc['decoder']);files+=[decoder,decoder.parent/'audit.json',decoder.parent/'_SUCCESS.json']
    with np.load(decoder) as f:
        centers=f['centers'];local=f['local_basis'];shared=f['shared_basis']
        assert local.shape[0]==len(centers) and local.shape[-1]==centers.shape[-1]
        assets={'D02':len(shared)>=64,'D03':local.shape[1]>=16}
        for b in [*local[:,:min(16,local.shape[1])],shared[:min(64,len(shared))]]:
            np.testing.assert_allclose(b.astype(float)@b.astype(float).T,np.eye(len(b)),atol=2e-5)
        permutation=derangement(len(centers),cfg['permutation_seed']).tolist()
    parent_inputs=json.loads((main/'inputs.json').read_text())
    by_id={r['sample_id']:r for r in parent_inputs if r['split']=='validation'}
    assert len(by_id)==64
    selected_ids=ordered(by_id,cfg['validation_salt'])[:32]
    reuse={};validation=[]
    for sid in selected_ids:
        row=dict(by_id[sid],cohort='validation',dataset='math');validation.append(row)
        paths={}
        for method in ['baseline','c1_1.0','c1_0.3']:
            p=main/'outputs'/sid/(method+'.json');record=json.loads(p.read_text())
            assert json.loads(p.with_suffix('.receipt.json').read_text())['record_sha256']==sha(p)
            assert record['split']=='validation' and record['plan_sha256']==sha(main/'plan.json')
            for rel,h in record['files'].items():
                dep=p.parent/rel;assert sha(dep)==h;files.append(dep)
            files.extend([p,p.with_suffix('.receipt.json')]);paths[method]=str(p)
        reuse[sid]=paths
    old_prompt_hashes=set();old_question_hashes=set();prior_ids=set();exclusion_sources=[]
    metadata=sorted(Path(pc['foundation']).glob('prefixes/shard_*/rows.parquet'))
    metadata.append(Path(pc['transfer'])/'rows.parquet')
    for p in metadata:
        frame=pd.read_parquet(p,columns=['sample_id','prompt_text'])
        prior_ids.update(frame.sample_id.astype(str));old_prompt_hashes.update(map(text_hash,frame.prompt_text))
        files.append(p);exclusion_sources.append(dict(path=str(p),rows=len(frame),sha256=sha(p)))
    prior_plan=Path(cfg['prior_confirmation'])/'collection_plan.json'
    prior=json.loads(prior_plan.read_text());files.append(prior_plan)
    for r in prior['selected']:
        prior_ids.add(r['sample_id']);old_prompt_hashes.add(text_hash(r['prompt_text']));old_question_hashes.add(text_hash(r['question']))
    exclusion_sources.append(dict(path=str(prior_plan),rows=len(prior['selected']),sha256=sha(prior_plan)))
    template=get_template('math','zot');assert template.hash==cfg['template_hash']
    dataset=load_dataset(cfg['dataset'],cfg['dataset_config'],split='train',revision=cfg['dataset_revision'])
    assert len(dataset)==7473
    pool=[];excluded=[];seen=set()
    for i,row in enumerate(dataset):
        sid=f'gsm8k_train_{i}';question=row['question'];qh=text_hash(question);prompt=template.format(problem=question)
        if sid in prior_ids or qh in old_question_hashes or text_hash(prompt) in old_prompt_hashes or qh in seen:
            excluded.append(sid);continue
        seen.add(qh)
        order=hashlib.sha256((cfg['transfer_salt']+'\0'+qh).encode()).hexdigest()
        pool.append((order,sid,dict(sample_id=sid,source_sample_idx=i,question=question,question_hash=qh,
             question_group=qh,prompt_text=prompt,ground_truth=row['answer'].rsplit('####',1)[1].strip(),
             dataset_answer=row['answer'],split='transfer',cohort='transfer',dataset='gsm8k')))
    assert len(pool)>=64
    transfer=[row for _,_,row in sorted(pool)[:64]]
    tok=AutoTokenizer.from_pretrained(pc['model'],revision=pc['revision'])
    for row in transfer:
        rendered=tok.apply_chat_template([dict(role='user',content=row['prompt_text'])],tokenize=False,add_generation_prompt=True)
        row.update(prompt_ids=tok(rendered,add_special_tokens=False).input_ids,model_input_text=rendered)
    assert not {r['sample_id'] for r in transfer}&prior_ids
    assert not {r['question_hash'] for r in transfer}&old_question_hashes
    # No primary-test records are opened. Tier selection is deferred to timing-only queue startup.
    freeze(root/'inputs.json',validation+transfer);freeze(root/'reuse.json',reuse)
    freeze(root/'exclusions.json',dict(sources=exclusion_sources,excluded_ids=excluded,prior_id_count=len(prior_ids),
        prior_prompt_hash_count=len(old_prompt_hashes),selected_question_hashes=[r['question_hash'] for r in transfer],
        scope='New project-held-out GSM8K train questions, excluding historical MATH5000, GSM8K test1319, and prior train64 confirmation. No claim about model pretraining contamination.'))
    variants=catalog()
    for c in variants:c['asset_available']=assets.get(c['name'],True)
    source_dir=Path(__file__).parent
    files += list(source_dir.glob('*projection_v2*.py'))
    files += [source_dir/n for n in ['evaluate_revision_completion.py','extract_revision_prefixes.py','revision_common.py',
        'revision_completion_common.py','revision_locality_common.py','check_revision_completion_hook.py']]
    files += [source_dir/'report_revision_completion.py']
    files += [Path(inspect.getfile(get_template)),root/'inputs.json',root/'reuse.json',root/'exclusions.json']
    files += list((source_dir.parent/'docs/projection-first-v2/request').glob('*'))
    inherited={k:pc[k] for k in ['model','revision','layer','max_new_tokens','decoder']}
    plan=provenance(dict(cfg,**inherited),sorted(set(files)))
    adopted=datetime.fromisoformat(cfg['adopted_utc']).timestamp()
    plan.update(adopted_unix=adopted,deadline_unix=adopted+24*3600,no_new_batch_unix=adopted+20*3600,
        conditions=variants,transfer_conditions=transfer_conditions(),permutation=permutation,
        decoder_shape=list(centers.shape),dataset_fingerprint=dataset._fingerprint,template=template.to_dict(),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        baseline_reuse='Exact immutable primary validation records for t16. D07/D08 require32 timing-matched baselines each within64-generation reserve.',
        protected_random_semantics='Three seeds of the SAME radial/energy-matched mechanism; average within question.',
        anchor='Exact protected decoder GMM centers, nearest-center Euclidean assignment; local residual SVD bases.',
        exploratory_scope='All ten variants use32 of the original64 validation questions; no new winner confirmation.',
        statistics='Frozen-primary statistics unchanged; new outcomes use question-paired bootstrap, all pointwise95%; no simultaneous-significance claim.',
        case_review='All repair/damage pairs exported with blinded method order; semantic review remains pending until actual inspection.')
    freeze(root/'plan.json',plan)
    write_json(root/'prepare_SUCCESS.json',dict(complete=True,plan_sha256=sha(root/'plan.json'),validation=32,transfer=64,
        available_variants=[c['name'] for c in variants if c['asset_available']],new_generation_ceiling=640))
    print(json.dumps(dict(prepared=True,validation=32,transfer=64,assets=assets,decoder_shape=centers.shape,
                         deadline_utc=datetime.fromtimestamp(plan['deadline_unix']).isoformat())))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);run(json.loads(Path(ap.parse_args().config).read_text()))
