from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_factor_common import conditions,key
from report_revision_fair_comparison import statistics,METRICS


def frame():
    rows=[]
    for sid,base in [('a',1.),('b',7.),('c',-2.)]:
        for c in conditions():
            value=base+(2 if c['method']=='fa' else 0)
            rows.append({'dataset':'math','split':'test','sample_id':sid,'method':key(c),
                         **{m:value for m in METRICS}})
    return pd.DataFrame(rows)


def test_primary_uses_paired_questions_and_keeps_both_endpoints():
    summary,pairs=statistics(frame())
    primary=[p for p in pairs if p['primary']]
    assert {p['metric'] for p in primary}=={'kl','delta_nll'}
    assert all(p['n']==3 and p['estimate']==p['low']==p['high']==2 for p in primary)
    assert len(summary)==19*5 and len(pairs)==3*6*5


def test_duplicate_questions_are_rejected_before_intervals():
    data=frame()
    with pytest.raises(ValueError,match='independent unit'):
        statistics(pd.concat([data,data.iloc[:1]],ignore_index=True))
