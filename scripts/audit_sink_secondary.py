"""Independent enumeration/mean checks; imports no functions from the analyzer."""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / 'results/sink-secondary-20260925'
src = ROOT / 'results/sink-followup-20260925'
s = json.loads((r / 'summary.json').read_text())
for name, expected in s['sources'].items():
    assert hashlib.sha256((src / name).read_bytes()).hexdigest() == expected
f = pd.read_parquet(src / 'step1_reencode.parquet')
f = f[f.arm.isin(['zero', 'hss', 'random'])].copy()
f['S'] = f.global_ids.map(lambda a: json.loads(a)[14] == 24)
f['D'] = (~f.boxed) | (f.n_tokens >= 2048)
dist, observed = {0: 1.}, 0
for sid, g in f.groupby('sample_id'):
    a, b = g.S.to_numpy(int), g.D.to_numpy(int)
    observed += sum((a[i]-a[j]) * (b[i]-b[j]) for i, j in itertools.combinations(range(3), 2))
    scores = Counter(sum((a[i]-a[j]) * (p[i]-p[j]) for i, j in itertools.combinations(range(3), 2))
                     for p in itertools.permutations(b))
    new = Counter()
    for k, v in dist.items():
        for j, c in scores.items():
            new[k+j] += v*c/6
    dist = new
p = min(1, 2 * min(sum(v for k, v in dist.items() if k <= observed),
                   sum(v for k, v in dist.items() if k >= observed)))
assert abs(p-s['behavior']['pooled']['association_test']['p_two_sided']) < 1e-12
assert observed == s['behavior']['pooled']['association_test']['statistic']
long = pd.read_parquet(src / 'step2_screen_long.parquet')
for group, col, key in [('sink', 'm_loop', 'sink_delta_vs_NONE'), ('normal', 'm_col_nll', 'normal_delta_vs_NONE')]:
    t = long[long['set'] == group].pivot(index='question_id', columns='condition', values=col).dropna()
    for candidate, row in s['likelihood']['candidates'].items():
        assert abs((t[candidate]-t.NONE).mean()-row[key]['mean']) < 1e-12
        for control, z in row['paired_controls'].items():
            assert abs((t[candidate]-t[control]).mean()-z[group+'_candidate_minus_control']['mean']) < 1e-12
t = long[long['set'] == 'sink'].pivot(index='question_id', columns='condition', values='m_loop').dropna()
controls = [c for c in t if c.startswith('CTRL_CLAMP_')]
x = np.column_stack([(t.C1-t[c]).to_numpy() for c in controls])
rng, samples = np.random.default_rng(811), []
for _ in range(40):
    samples.append(x[rng.integers(0, len(x), (500, len(x)))].mean(1))
intervals = np.quantile(np.concatenate(samples), [.000625, .999375], axis=0)
assert (intervals[0] > 0).all()
receipt = dict(passed=True, conditional_exact_p_independent=p,
               table_means_all_candidates_and_controls_verified=True,
               C1_all10_separately_resampled_CI_positive=True,
               independent_bootstrap_B=20000, minimum_lower_bound=float(intervals[0].min()),
               summary_sha256=hashlib.sha256((r / 'summary.json').read_bytes()).hexdigest())
(r / 'independent_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
