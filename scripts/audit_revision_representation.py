"""Predeclared last-vs-window comparisons on identical historical-test questions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import write_json
from revision_statistics import paired_auc_ci


def run(root):
    rows=[]
    for layer in (14,28):
        for prefix in (0,16,64):
            a=pd.read_parquet(root/'geometry'/f'p{prefix}_l{layer}_mean16'/'prediction_per_question.parquet')
            b=pd.read_parquet(root/'geometry'/f'p{prefix}_l{layer}_last'/'prediction_per_question.parquet')
            a=a.loc[a.split.eq('test')].set_index('sample_id')
            b=b.loc[b.split.eq('test')].set_index('sample_id')
            common=sorted(a.index.intersection(b.index))
            a,b=a.loc[common],b.loc[common]
            np.testing.assert_array_equal(a.failure,b.failure)
            for method in ('linear_probe','state_nb'):
                row={'block':layer,'prefix':prefix,'comparison':f'mean16 versus last / {method}',
                     **paired_auc_ci(a.failure,a[method],b[method])}
                rows.append(row)
            rows.append({'block':layer,'prefix':prefix,'comparison':'mean16 nuisance+state versus nuisance',
                         **paired_auc_ci(a.failure,a.nuisance_plus_state,a.nuisance)})
    summary={'scope':'Qwen2-7B raw pre-norm, MATH reused historical test: exploratory, not causal',
             'split':'3011 train / 1003 validation / 986 historical test; comparisons join question IDs',
             'readouts':'Clusters fit without labels; NB/LR use train correctness labels. C selected on validation only.',
             'nuisance':'category/difficulty/prompt length/representation norm/current entropy and margin; no future answer length',
             'cautions':['pointwise intervals; no family-wise correction','Different positions/window lengths contain different amounts of information',
                         'Prediction does not show that mean is a computational state',
                         'Four-position token-fit intervention decoders are superseded by full-window v2 before causal testing'],
             'comparisons':rows}
    write_json(root/'representation_audit.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
