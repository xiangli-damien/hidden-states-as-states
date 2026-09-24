"""Independent raw-label paired accuracy/transition/selection audit."""
import argparse,json
from pathlib import Path
import numpy as np
from revision_common import sha,write_json


def run(root):
    cfg=json.loads((root/'plan.json').read_text())['config'];s=json.loads((root/'steering_summary.json').read_text())
    assert s['complete'];arrays={};counts=0
    for stage in ['validation']+(['test'] if s['selection']['run_test'] else []):
        rc=json.loads((root/f'{stage}_SUCCESS.json').read_text());data={}
        audit=json.loads((root/f'{stage}_audit.json').read_text())
        assert audit['complete'] and audit['stage_receipt_sha256']==sha(root/f'{stage}_SUCCESS.json')
        for rel,h in rc['records'].items():
            assert sha(root/rel)==h;r=json.loads((root/rel).read_text())
            data.setdefault(r['condition']['name'],{})[r['sample_id']]=r
        ids=sorted(data['baseline']);base=np.array([data['baseline'][k]['correct'] for k in ids],float)
        stats={name:np.array([items[k]['correct'] for k in ids],float) for name,items in data.items()}
        seeds=[name for name in data if name.startswith('random_')]
        if seeds:assert len(seeds)==3;stats['random_seed_average']=np.mean([stats[n] for n in sorted(seeds)],axis=0)
        arrays[stage]=stats
        for row in [r for r in s['rows'] if r['stage']==stage]:
            pred=stats[row['method']];delta=pred-base
            for key,values in [('accuracy',pred),('delta_accuracy',delta)]:
                rng=np.random.default_rng(cfg['bootstrap_seed']);samples=[]
                for _ in range(cfg['bootstrap_draws']):samples.append(np.mean(values[rng.choice(len(ids),len(ids),replace=True)]))
                np.testing.assert_allclose(row[key]['estimate'],np.mean(values),rtol=0,atol=1e-15)
                np.testing.assert_allclose(row[key]['ci95'],np.quantile(samples,[.025,.975]),rtol=0,atol=1e-15)
            assert row['n']==len(ids)
            np.testing.assert_allclose(row['wrong_to_correct_expected'],np.dot(1-base,pred))
            np.testing.assert_allclose(row['correct_to_wrong_expected'],np.dot(base,1-pred));counts+=1
        if stage=='validation':
            aggregates=[]
            for row in s['selection']['validation']:
                name=row['condition']['name'];items=data[name]
                net=int(sum(int(items[k]['correct'])-int(data['baseline'][k]['correct']) for k in ids))
                parse=int(sum(int(items[k]['parse_failed'])-int(data['baseline'][k]['parse_failed']) for k in ids))
                assert row['net_correct']==net and row['parse_failure_increase']==parse
                if net>0 and parse<=2:
                    energy=np.mean([items[k]['geometry']['energy'] for k in ids]);c=row['condition']
                    aggregates.append(((-net,energy,c['alpha'],c['family']!='c1',name),c))
            if s['selection']['run_test']:
                assert s['selection']['selected']==min(aggregates,key=lambda x:x[0])[1]
            elif s['selection']['reason']=='no_candidate_with_positive_net_gain_and_format_gate':assert not aggregates
    for row in s['paired_controls']:
        values=arrays[row['stage']][row['method']]-arrays[row['stage']][row['control']]
        rng=np.random.default_rng(cfg['bootstrap_seed'])
        sims=[np.mean(values[rng.choice(len(values),len(values),replace=True)]) for _ in range(cfg['bootstrap_draws'])]
        np.testing.assert_allclose(row['accuracy_difference']['estimate'],np.mean(values),rtol=0,atol=1e-15)
        np.testing.assert_allclose(row['accuracy_difference']['ci95'],np.quantile(sims,[.025,.975]),rtol=0,atol=1e-15)
    write_json(root/'steering_statistics_audit.json',{'complete':True,'summary_rows':counts,'paired_rows':len(s['paired_controls']),
        'source_summary_sha256':sha(root/'steering_summary.json'),'code_sha256':sha(Path(__file__)),
        'independent':'Raw JSON labels -> per-question random means -> paired bootstrap, validation gates and selected policy'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)
