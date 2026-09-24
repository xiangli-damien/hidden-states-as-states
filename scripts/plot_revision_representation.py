"""Export audited, paired question-level representation comparisons."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(root):
    audit=json.loads((root/'representation_audit.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors={14:'#007f86',28:'#b65320'}
    for axis,comparison,title in [(axes[0],'mean16 versus last / linear_probe','Window mean versus last token'),
                                  (axes[1],'mean16 nuisance+state versus nuisance','Discrete-state increment beyond controls')]:
        for layer in (14,28):
            rows=sorted([r for r in audit['comparisons'] if r['block']==layer and r['comparison']==comparison],key=lambda r:r['prefix'])
            x=np.arange(3)+(-.08 if layer==14 else .08)
            y=np.array([r['delta'] for r in rows]);low=np.array([r['ci95'][0] for r in rows]);high=np.array([r['ci95'][1] for r in rows])
            axis.errorbar(x,y,yerr=[y-low,high-y],marker='o',capsize=4,lw=1.5,color=colors[layer],label=f'Block {layer}')
        axis.axhline(0,color='.4',ls='--',lw=1);axis.set_xticks(range(3),['Prompt','Generated 16','Generated 64'])
        axis.set_ylabel('Paired difference in failure AUROC');axis.set_title(title,fontsize=11)
        axis.grid(axis='y',alpha=.2);axis.legend(frameon=False)
    fig.suptitle('Qwen2-7B MATH · raw pre-norm activations · same 986 historical-test questions',fontsize=11)
    dest=root/'report';dest.mkdir(exist_ok=True,parents=True)
    fig.savefig(dest/'representation_comparison.png',dpi=180)
    fig.savefig(dest/'representation_comparison.pdf')
    plt.close(fig)
    (dest/'representation_figure_caption.md').write_text(
        'Paired differences use 2,000 question-level bootstrap draws with exact score ties. '
        'Intervals are pointwise and unadjusted for multiple comparisons. Historical MATH test has been explored; '
        'these are exploratory prediction results, not causal evidence. Prompt means use the last 16 rendered chat-input tokens; '
        'generation means use the last 16 consumed generated tokens. Both continuous vectors have 3,584 dimensions. '
        'Clustering uses raw values; supervised linear readouts fit scaling/regularization on train/validation only. '
        'Controls include task type, difficulty, prompt length, representation norm, current next-token entropy and margin; '
        'no future completion length enters these controls.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
