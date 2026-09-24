"""Question-paired v2 results, all flips, and descriptive saved-vector geometry."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from projection_v2_common import paired_interval,geometry
from revision_common import sha,write_json


def load_record(path):
    path=Path(path);r=json.loads(path.read_text())
    assert json.loads(path.with_suffix('.receipt.json').read_text())['record_sha256']==sha(path)
    for rel,h in r['files'].items():assert sha(path.parent/rel)==h
    return dict(r,record_path=str(path),record_sha256=sha(path))


def load_batch(root,batch):
    audit=json.loads((root/f'{batch}_audit.json').read_text())
    rc=json.loads((root/f'{batch}_SUCCESS.json').read_text())
    assert audit['complete'] and audit['source_receipt_sha256']==sha(root/f'{batch}_SUCCESS.json')
    out=[]
    for p,h in rc['records'].items():
        assert sha(root/p)==h;out.append(load_record(root/p))
    return out


def run(root):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];main=Path(cfg['protected_primary'])
    main_audit=json.loads((main/'steering_statistics_audit.json').read_text());primary=json.loads((main/'steering_summary.json').read_text())
    assert main_audit['complete'] and main_audit['source_summary_sha256']==sha(main/'steering_summary.json')
    tier=json.loads((root/'tier.json').read_text());reuse=json.loads((root/'reuse.json').read_text())
    raw=[];comparisons=[];case_pairs=[]
    def contrast(group,method,control,a,b):
        aa={r['sample_id']:r for r in a};bb={r['sample_id']:r for r in b}
        assert set(aa)==set(bb) and len(aa)==len(a)==len(b)
        ids=sorted(aa);x=np.array([aa[s]['correct'] for s in ids],float);y=np.array([bb[s]['correct'] for s in ids],float)
        item=dict(group=group,method=method,control=control,n=len(ids),accuracy=float(x.mean()),control_accuracy=float(y.mean()),
            delta_accuracy=paired_interval(x-y),wrong_to_correct=int(((x==1)&(y==0)).sum()),correct_to_wrong=int(((x==0)&(y==1)).sum()),
            parse_failure_rate=float(np.mean([aa[s]['parse_failed'] for s in ids])),truncation_rate=float(np.mean([aa[s]['finish_reason']=='length' for s in ids])),
            mean_length=float(np.mean([aa[s]['length'] for s in ids])),mean_seconds=float(np.mean([aa[s]['seconds'] for s in ids])),
            intervention_coverage=float(np.mean([aa[s]['geometry']['modified_tokens']>0 for s in ids])),
            per_question=[dict(sample_id=s,method_record=aa[s]['record_path'],control_record=bb[s]['record_path']) for s in ids])
        comparisons.append(item)
        for sid in ids:
            if aa[sid]['correct']!=bb[sid]['correct']:
                key=hashlib.sha256(f'{group}/{method}/{control}/{sid}'.encode()).hexdigest()[:16]
                flip=int(key[-1],16)%2;left,right=(aa[sid],bb[sid]) if flip else (bb[sid],aa[sid])
                case_pairs.append(dict(case_id=key,group=group,method=method,control=control,sample_id=sid,
                    effect='repair' if aa[sid]['correct'] else 'damage',left_record=left['record_path'],right_record=right['record_path'],
                    method_side='left' if flip else 'right',review_status='pending_blinded_semantic_review',
                    semantic_category=None,notes=None,
                    objective_flags=dict(parse_failure_changed=aa[sid]['parse_failed']!=bb[sid]['parse_failed'],
                        truncation_changed=(aa[sid]['finish_reason']=='length')!=(bb[sid]['finish_reason']=='length'),
                        normalized_answer_changed=aa[sid]['normalized_answer']!=bb[sid]['normalized_answer'])))
    # Protected test metrics are imported unchanged. Raw paths permit re-audit and case review.
    test_rc=json.loads((main/'test_SUCCESS.json').read_text());test_records=[]
    for rel,h in test_rc['records'].items():
        assert sha(main/rel)==h;test_records.append(load_record(main/rel))
    raw+=test_records
    contrast('MATH_primary256_supplement','local8','baseline',
        [r for r in test_records if r['condition']['name']=='c1_1.0'],[r for r in test_records if r['condition']['name']=='baseline'])
    original_baseline=[load_record(paths['baseline']) for paths in reuse.values()]
    original_local=[load_record(paths['c1_1.0']) for paths in reuse.values()]
    raw+=original_baseline+original_local
    # All64 selection-set repairs/damages are available for blinded review, separate from confirmation.
    val_rc=json.loads((main/'validation_SUCCESS.json').read_text());v=[]
    for rel,h in val_rc['records'].items():
        if Path(rel).stem in ['baseline','c1_1.0']:
            assert sha(main/rel)==h;v.append(load_record(main/rel))
    contrast('MATH_original_validation64','local8','baseline',[r for r in v if r['condition']['name']=='c1_1.0'],[r for r in v if r['condition']['name']=='baseline'])
    raw+=v
    batches=[]
    for batch in ['transfer','D01','D02','D03','D05','D06','D04','D07','D08','D09','D10']:
        if not (root/f'{batch}_audit.json').exists():continue
        data=load_batch(root,batch);raw+=data;batches.append(batch)
        if batch=='transfer':
            base=[r for r in data if r['condition']['name']=='baseline'];loc=[r for r in data if r['condition']['name']=='local8']
            for method in tier['transfer_conditions']:
                if method!='baseline':contrast('GSM8K_frozen_transfer',method,'baseline',[r for r in data if r['condition']['name']==method],base)
            for control in ['shared8','wrong_local8']:
                cc=[r for r in data if r['condition']['name']==control]
                if cc:contrast('GSM8K_frozen_transfer','local8',control,loc,cc)
        else:
            changed=[r for r in data if r['condition']['name']==batch]
            base=[r for r in data if r['condition']['name']=='baseline'] if batch in ['D07','D08'] else original_baseline
            contrast('MATH_exploratory32',batch,'baseline_same_trigger' if batch in ['D07','D08'] else 'baseline',changed,base)
            contrast('MATH_exploratory32',batch,'original_local8_t16',changed,original_local)
            if batch=='D04':
                positive=[load_record(paths['c1_0.3']) for paths in reuse.values()];raw+=positive
                contrast('MATH_exploratory32','D04','original_alpha_plus0.3',changed,positive)
    # Diagnostics do not define a new correctness gate or change any sample membership.
    with np.load(cfg['decoder']) as f:decoder={k:f[k] for k in ['centers','local_basis','shared_basis']}
    counts=json.loads((main/'support.json').read_text())['question_count'];flat=[]
    unique={r['record_path']:r for r in raw}
    for r in unique.values():
        g=r['geometry'];p=Path(r['record_path']).with_suffix('.npz')
        if p.exists():
            with np.load(p) as f:g=geometry(f['before'],f['actual'],decoder,counts)
        flat.append(dict(sample_id=r['sample_id'],split=r['split'],method=r['condition']['name'],record_path=r['record_path'],
            correct=r['correct'],length=r['length'],parse_failed=r['parse_failed'],finish_reason=r['finish_reason'],
            energy=g['energy'],region_retention=float(np.mean(g['region_retained'])) if g.get('region_retained') else None,
            all_regions_retained=bool(all(g['region_retained'])) if g.get('region_retained') else None,
            offspace_fraction=float(np.mean(g['offspace_fraction'])) if g.get('offspace_fraction') else None,
            relative_change=float(np.mean(g['relative_change'])) if g.get('relative_change') else None))
    pd.DataFrame(flat).to_csv(root/'geometry_behavior.csv',index=False)
    write_json(root/'case_review_key.json',case_pairs)
    write_json(root/'records.json',[dict(path=p,sha256=r['record_sha256']) for p,r in unique.items()])
    write_json(root/'summary.json',dict(complete=True,tier=tier,batches=batches,comparisons=comparisons,
        primary_rows=[r for r in primary['rows'] if r['stage']=='test'],primary_paired=primary['paired_controls'],
        protected_primary_summary_sha256=sha(main/'steering_summary.json'),case_pairs=len(case_pairs),
        limitations=['Primary statistics imported unchanged; supplemental bootstrap shown separately.',
          'All v2 intervals are pointwise question bootstrap; no simultaneous significance claim.',
          'Original64 and new32 MATH analyses are exploratory, not new independent confirmation.',
          'Transfer questions are new within project from official GSM8K train; no claim about model pretraining.',
          'Blinded semantic review is pending; objective parse/truncation flags are not reasoning-error labels.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)
