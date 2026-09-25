"""Publish compact, source-verified matched-control results and a scientific plot."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision_common import sha

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results/sink-energy-matched-20260925-v3'
dest = ROOT / 'docs/sink-energy-matched-20260925'
dest.mkdir(exist_ok=True)
s = json.loads((source/'summary.json').read_text())
a = json.loads((source/'audit_SUCCESS.json').read_text())
assert a['passed'] and a['summary_sha256'] == sha(source/'summary.json')
for name in ['summary.json', 'audit_SUCCESS.json', 'independent_local_audit.json']:
    shutil.copy2(source/name, dest/name)
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.4,4.7),sharey=True)
labels=['Sink-axis clamp − NONE','Orthogonal-control mean − NONE','Clamp − control mean']
for ax,group,title in zip(axes,['sink','normal'],['Sink reference suffixes (n=96)','Normal reference responses (n=100)']):
    g=s['groups'][group]
    rows=[g['deltas_vs_NONE']['C1'],g['control_mean_vs_NONE'],g['c1_minus_control_mean']]
    for i,row in enumerate(rows):
        m,(lo,hi)=row['mean'],row['ci']
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=['#166A80','#AB8334','#5A659A'][i],lw=2,capsize=5,markersize=7)
    ax.axvline(0,color='#939AA1',ls='--',lw=1)
    ax.grid(axis='x',alpha=.12)
    ax.set_title(title)
    ax.set_xlabel('Mean NLL difference (nats/token)')
axes[0].set_yticks(range(3),labels)
axes[0].set_ylim(2.6,-.6)
axes[0].set_xlim(-.005,.11)
axes[1].set_xlim(-.0011,.0007)
axes[1].set_xticks([-.001,-.0005,0,.0005],['−0.0010','−0.0005','0','+0.0005'])
fig.suptitle('Per-token energy-matched controls: paired 98.75% intervals',fontsize=13)
fig.text(.5,.035,'Identical active positions; maximum response-energy mismatch 0.0217%. Fixed-token likelihood, not free-generation recovery.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.08,1,.96])
fig.savefig(dest/'effects.png',dpi=175)
plt.close(fig)
