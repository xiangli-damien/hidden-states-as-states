"""Publish lightweight tables and a figure from the sealed CPU secondary analysis."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / 'results/sink-secondary-20260925'
o = ROOT / 'docs/sink-secondary-20260925'
o.mkdir(exist_ok=True)
for name in ['summary.json', 'geometry_summary.json', 'independent_audit.json']:
    shutil.copy2(r / name, o / name)
s = json.loads((r / 'summary.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
names = ['C0: fixed direction', 'C1: sink-axis clamp', 'C2: type-specific direction', 'C3: chart-projected direction']
for ax, key, title in zip(axes, ['sink_minus_control_mean', 'normal_minus_control_mean'],
                          ['Sink reference suffixes (96 questions)', 'Normal reference responses (100 questions)']):
    for i, c in enumerate(['C0', 'C1', 'C2', 'C3']):
        d = s['likelihood']['candidates'][c][key]
        m, (lo, hi) = d['mean'], d['ci']
        ax.errorbar(m, i, xerr=[[m-lo], [hi-m]], fmt='o',
                    color='#166A80' if c == 'C1' else '#7D8995', capsize=4, lw=2, markersize=7)
        ax.text(.97, i+.18, f'{m:+.6f}', transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=9)
    ax.axvline(0, color='#B3B8BE', ls='--', lw=1)
    ax.set_title(title)
    ax.set_xlabel('Candidate minus mean matched-form control\nMean NLL difference (nats/token)')
    ax.grid(axis='x', alpha=.15)
axes[0].set_yticks(range(4), names)
axes[0].set_ylim(3.7, -.5)
axes[0].set_xlim(-.003, .115)
axes[1].set_xlim(-.0011, .0010)
axes[1].set_xticks([-.001, -.0005, 0, .0005, .001], ['−0.0010', '−0.0005', '0', '+0.0005', '+0.0010'])
fig.suptitle('Post-hoc likelihood effects: paired 98.75% intervals, four-candidate correction', fontsize=13)
fig.text(.5, .018, 'Actual update magnitude was not matched. These are fixed-sequence likelihood effects, not free-generation success.', ha='center', fontsize=10)
fig.tight_layout(rect=[0, .07, 1, .96])
fig.savefig(o / 'likelihood-effects.png', dpi=170)
plt.close(fig)
