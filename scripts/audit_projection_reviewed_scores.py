"""Independent provenance/label-join/paired-statistic audit; no semantic judging."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(r): return hashlib.sha256((r['prompt_text']+'\0'+str(r['ground_truth'])+'\0'+r['response_text']).encode()).hexdigest()


def run(root, primary, out):
    s=read(out/'summary.json');scored=root/'scoring_sensitivity_v2'
    assert s['complete'] and not s['full_semantic_rescoring_complete'] and not s['independent_human_review']
    for p,h in {**s['input_hashes'],**s['source_hashes']}.items(): assert sha(p)==h,p
    for p,h in read(out/'_SUCCESS.json')['files'].items(): assert sha(out/p)==h,p
    manifest=read(root/'records.json');raw={}
    for m in manifest:
        p=Path(m['path']);prefix=Path('/lambda/nfs/dami/hss')
        if p.is_relative_to(prefix/root.name): p=root/p.relative_to(prefix/root.name)
        elif p.is_relative_to(prefix/primary.name): p=primary/p.relative_to(prefix/primary.name)
        assert sha(p)==m['sha256'];raw[m['path']]=read(p)
    assert len(raw)==len(manifest)
    known={}
    for n in read(root/'semantic_review_v2/summary.json')['rows']:
        for side,letter in [('left','A'),('right','B')]:
            aid=ident(raw[n[side+'_record']]);v=n['final_answer_correct_'+letter]
            assert aid not in known or known[aid]==v
            known[aid]=v
    prior=set(known)
    notes=read(scored/'semantic_notes_blinded.json');packet=read(scored/'review_packet.json')
    assert notes['packet_sha256']==sha(scored/'review_packet.json') and notes['complete']
    assert set(notes['notes'])=={x['answer_id'] for x in packet['answers']}
    for i,a in enumerate(packet['answers']):
        aid=hashlib.sha256((a['prompt']+'\0'+str(a['reference'])+'\0'+a['response']).encode()).hexdigest()
        assert aid==a['answer_id'];n=notes['notes'][aid]
        assert n['full_answer_read'] and n['packet_index']==i and isinstance(n['final_answer_correct'],bool)
        assert aid not in known or known[aid]==n['final_answer_correct']
        known[aid]=n['final_answer_correct']
    rows=[json.loads(x) for x in (out/'scores.jsonl').read_text().splitlines()]
    bypath={r['path']:r for r in rows}
    assert set(bypath)==set(raw) and len(rows)==len(bypath)==s['records']
    v1={r['path']:r for r in map(json.loads,(root/'scoring_sensitivity_v1/scores.jsonl').read_text().splitlines())}
    v2={r['path']:r for r in map(json.loads,(scored/'scores.jsonl').read_text().splitlines())}
    assert set(v1)==set(v2)==set(raw)
    for p,r in bypath.items():
        assert all(r[k]==v for k,v in v2[p].items())
        aid=ident(raw[p]);assert aid==r['answer_id']
        assert r['original_correct']==raw[p]['correct'] and r['v1_correct']==v1[p]['new_correct']
        assert r['reviewed_correct']==(known[aid] if aid in known else v2[p]['new_correct'])
        assert r['label_source']==('sealed_AI_full_answer_reading' if aid in known else 'automatic_v2_unreviewed')
    changed=[r for r in rows if r['original_correct']!=r['new_correct']]
    assert all(r['answer_id'] in known for r in changed)
    assert {r['answer_id'] for r in changed if r['answer_id'] not in prior}==set(notes['notes'])
    nr=sum(r['answer_id'] in known for r in rows)
    assert nr==s['AI_reviewed_records'] and len(known)==s['AI_reviewed_unique_answers']
    assert len(rows)-nr==s['unreviewed_records']
    assert len(changed)==s['changed_automatic_records']==s['changed_automatic_records_with_reading']
    assert len(notes['notes'])==s['new_full_answers_read']
    assert sum(r['original_correct']!=r['reviewed_correct'] for r in rows)==s['labels_changed_vs_original']
    dis=[r for r in rows if r['answer_id'] in known and r['new_correct']!=r['reviewed_correct']]
    assert dis==read(out/'v2_disagreements.json') and len(dis)==s['v2_disagreement_records']
    assert {k:v['correct'] for k,v in read(out/'readings.json').items()}==known
    original={(c['group'],c['method'],c['control']):c for c in read(root/'summary.json')['comparisons']}
    assert len(s['comparisons'])==len(original)+2
    assert len({(c['group'],c['method'],c['control']) for c in s['comparisons']})==len(s['comparisons'])
    for c in s['comparisons']:
        key=c['group'],c['method'],c['control']
        if key in original:
            expected=[dict(method_record=p['method_record'],control_records=[p['control_record']]) for p in original[key]['per_question']]
        else:
            assert key in [('MATH_primary256_supplement','local8','shared8'),('MATH_primary256_supplement','local8','random_mean3')]
            test={}
            for r in rows:
                if r['split']=='test':test.setdefault(r['sample_id'],{})[r['condition']]=r['path']
            controls=['shared8'] if c['control']=='shared8' else ['random_42','random_137','random_271']
            expected=[dict(method_record=v['c1_1.0'],control_records=[v[k] for k in controls]) for _,v in sorted(test.items())]
        assert expected==c['pairs']
        assert c['fully_AI_read_pairs']==sum(all(bypath[p]['answer_id'] in known for p in [x['method_record']]+x['control_records']) for x in expected)
        for version,field in [('original','original_correct'),('v1','v1_correct'),('v2','new_correct'),('partial_AI_review','reviewed_correct')]:
            a=np.array([bypath[p['method_record']][field] for p in expected],float)
            b=np.array([np.mean([bypath[x][field] for x in p['control_records']]) for p in expected])
            z=c[version];n=len(a);d=a-b
            assert z['n']==n
            np.testing.assert_allclose([z['method_accuracy'],z['control_accuracy'],z['delta']],[a.mean(),b.mean(),d.mean()],atol=1e-14,rtol=0)
            indices=np.random.default_rng(9242026).integers(0,n,(2000,n))
            np.testing.assert_allclose(z['ci95'],np.quantile(d[indices].mean(axis=1),[.025,.975]),atol=1e-14,rtol=0)
            if c['control']=='random_mean3':assert 'repair' not in z and 'damage' not in z
            else:
                assert z['repair']==int(((a==1)&(b==0)).sum())
                assert z['damage']==int(((a==0)&(b==1)).sum())
    receipt=dict(complete=True,records=len(rows),comparisons=len(s['comparisons']),scoring_versions=4,
                 provenance_and_uniform_joins_verified=True,paired_statistics_recomputed=True,
                 full_semantic_correctness_verified=False,summary_sha256=sha(out/'summary.json'),source_sha256=sha(__file__))
    with (out/'audit.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.primary_root,a.output)
