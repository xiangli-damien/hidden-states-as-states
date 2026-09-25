"""Build small, audited result-only report; never interprets partial outcomes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from revision_common import sha

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/sink-next-20260925'
ASSETS=REPO/'docs/sink-next-20260925'
REPORT=REPO/'docs/sink-next-results-20260925.zh-CN.md'


def f(x):return f'{x:.5f}'
def effect(x,ci='ci'):
    return f"{f(x['mean'])} [{f(x[ci][0])}, {f(x[ci][1])}]"


def run():
    summaries={};ASSETS.mkdir(parents=True,exist_ok=True)
    for i in [1,2,3,4]:
        p=ROOT/f'exp{i}'
        if not (p/'audit_SUCCESS.json').exists():continue
        a=json.loads((p/'audit_SUCCESS.json').read_text());assert a['passed'] and sha(p/'summary.json')==a['summary_sha256']
        summaries[i]=json.loads((p/'summary.json').read_text())
        dest=ASSETS/f'exp{i}';dest.mkdir(parents=True,exist_ok=True)
        for n in ['summary.json','audit_SUCCESS.json','FREEZE.json','independent_local_audit.json']:
            if (p/n).exists():shutil.copy2(p/n,dest/n)
    now=datetime.now(timezone.utc).isoformat()
    lines=['# HSS sink 推广实验：结果',f'\n更新：{now}。只展示完整、通过原始记录审核的实验。',
        '\n[冻结协议](sink-next-protocol-20260925.zh-CN.md) · [此前 v3 能量匹配结果](sink-energy-matched-results-20260925.zh-CN.md)',
        '\n**性质：探索性延伸，沿用此前已经检查过的问题；不是新样本确认实验。** NLL 单位为 nats/token，正差表示模型对既定参考续写的支持下降。']
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.2),layout='constrained')
    def plot(ax,x,est,lo,hi,label=None,connect=False):
        ax.errorbar(x,est,yerr=[np.array(est)-lo,np.array(hi)-est],fmt='o-' if connect else 'o',capsize=3,label=label)
        ax.axhline(0,color='#777',lw=.8);ax.grid(axis='y',alpha=.2)
    if 1 in summaries:
        s=summaries[1]['summaries']['matched'];g=s['groups'];c=s['primary_paired']
        lines += ['\n## 1. 同题型、同难度',
            f"\n{s['matching']['exact']} 对精确匹配；放宽一级 {s['matching']['relaxed']} 对；未匹配 {len(s['matching']['unmatched'])} 条。两组都使用 half_sentence 后半段窗口。",
            '\n| 量 | 均值 [95% CI] |\n|---|---:|',
            f"| Sink：C1−NONE | {effect(g['sink']['c1_vs_none'])} |",
            f"| 匹配非 sink：C1−NONE | {effect(g['matched']['c1_vs_none'])} |",
            f"| 主终点：匹配对差 | **{effect(c)}** |",
            '\n| 组 | 越界 token 比例（逐题均值） | 正超出量均值 | 实际更新范数均值 |\n|---|---:|---:|---:|']
        for row in summaries[1]['geometry']:
            lines += [f"| {row['set']} | {row['exceedance_fraction']:.2%} | {f(row['mean_positive_excess'])} | {f(row['mean_norm'])} |"]
        lines += ['\n结果超出已匹配的题型和难度。仍可能存在长度、回答内容等差异；两组自然截断的实际剂量不同，所以不把本项单独解释为完全排除题目因素或证明生成模式。',
            '\n本项 sink NLL 与 v3 的数值不同，是因为按预定规格统一改为 half_sentence，而 v3 包含 first_repeat 窗口。',
            f"\nEnglish: With problem type and difficulty matched in {c['n']} independent pairs, the sink-minus-control difference in the clamp-induced NLL increase was {c['mean']:.4f} nats/token (95% CI [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}])."]
        vals=[g['sink']['c1_vs_none'],g['matched']['c1_vs_none'],c]
        plot(axs[0],range(3),[v['mean'] for v in vals],[v['ci'][0] for v in vals],[v['ci'][1] for v in vals]);axs[0].set_xticks(range(3),['Sink','Matched','Paired gap']);axs[0].set_title('Type / level matching: 95% CI');axs[0].set_ylabel('NLL difference (nats/token)')
    else:axs[0].text(.5,.5,'Experiment 1 pending',ha='center',transform=axs[0].transAxes)
    if 2 in summaries:
        s=summaries[2]
        lines += ['\n## 2. 深度扫描','\n主列为 sink 的 C1−10 个对照均值；六个新层区间为 99.1667%（Bonferroni 6）。normal 列为描述性95%区间。',
            '\n| Index | Sink 主对比 [校正 CI] | Normal C1−NONE [95% CI] | Normal C1−对照 [95% CI] | cos(axis, axis14) |\n|---|---:|---:|---:|---:|']
        xx=[];vv=[]
        for name,v in sorted(s['summaries'].items(),key=lambda x:x[1]['layer']):
            c=v['groups']['sink']['contrast'];n=v['groups']['normal']['contrast'];xx.append(v['layer']);vv.append(c)
            lines += [f"| {v['layer']} | {effect(c)} | {effect(v['groups']['normal']['c1_vs_none'])} | {effect(n)} | {f(v['cosine_axis14'])} |"]
        plot(axs[1],xx,[v['mean'] for v in vv],[v['ci'][0] for v in vv],[v['ci'][1] for v in vv],label='New layers: 99.1667% CI',connect=True)
        prior=REPO/'docs/sink-energy-matched-20260925/summary.json'
        if prior.exists():
            v=json.loads(prior.read_text())['groups']['sink']['c1_minus_control_mean']
            axs[1].errorbar([14],[v['mean']],yerr=[[v['mean']-v['ci'][0]],[v['ci'][1]-v['mean']]],fmt='s',color='#bc6431',capsize=3,label='Index 14 (prior): 98.75% CI')
            lines += [f"\nIndex14 复用 v3：{effect(v)}，98.75% CI；未重新纳入六个新层的校正。"]
        axs[1].set_xlabel('Hidden-state index');axs[1].set_title('Depth: C1 minus controls');axs[1].legend(fontsize=8)
        lines += ['\n| Index | Sink 越界比例 | Normal 越界比例 | Sink 实际更新范数 | Normal 实际更新范数 |\n|---|---:|---:|---:|---:|']
        for name,v in sorted(s['summaries'].items(),key=lambda x:x[1]['layer']):
            geo={g['set']:g for g in s['geometry'] if g['spec']==name and g['condition']=='C1'}
            lines += [f"| {v['layer']} | {geo['sink']['exceedance_fraction']:.2%} | {geo['normal']['exceedance_fraction']:.2%} | {f(geo['sink']['mean_norm'])} | {f(geo['normal']['mean_norm'])} |"]
        lines += ['\n每层单独干预，同一评估人群和窗口。完整10个对照保存在 JSON 中。层间效应曲线不是等剂量比较，不能直接称最大效应层为最重要层。']
    else:
        axs[1].set_axis_off();axs[1].text(.5,.5,'Depth scan pending audit',ha='center',color='#666',transform=axs[1].transAxes)
    if 3 in summaries:
        lines += ['\n## 3. 四个其他高占用状态','\n主列为 in-state 的 C1−对照均值，98.75%区间（Bonferroni 4）；in−out为次要95%区间，两个组独立重采样。',
            '\n| 状态 | 训练人数 | 训练正确率 | 主导题型 | In-state 主对比 | Out-state 对比 | In−out 差 |\n|---|---:|---:|---|---:|---:|---:|']
        xx=[];vv=[]
        for name,v in summaries[3]['summaries'].items():
            c=v['groups']['in']['contrast'];xx.append(v['cluster']);vv.append(c);types=v['training_types'];dominant=max(types,key=types.get)
            lines += [f"| {v['cluster']} | {v['training_count']} | {v['training_correctness']:.1%} | {dominant} | {effect(c)} | {effect(v['groups']['out']['contrast'])} | {effect(v['in_minus_out'])} |"]
        plot(axs[2],range(len(xx)),[v['mean'] for v in vv],[v['ci'][0] for v in vv],[v['ci'][1] for v in vv]);axs[2].set_xticks(range(len(xx)),[str(x) for x in xx]);axs[2].set_xlabel('Local cluster ID');axs[2].set_title('Other states: 98.75% CI')
        lines += ['\n只检验预选的四个高占用状态。in-state 显著而 in−out 不显著时，不能声称状态特异；多数成立也不能外推所有状态。']
    else:
        axs[2].set_axis_off();axs[2].text(.5,.5,'Other states pending audit',ha='center',color='#666',transform=axs[2].transAxes)
        lines += ['\n## 3. 四个其他高占用状态','\n已按训练集占用人数预选，队列运行中；完整审核结果尚未同步，不报告部分效应。每个状态的 in-B 最多100条，out-B最多100条，不重复采样补足。']
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(ASSETS/'effects.png',dpi=170);fig.savefig(ASSETS/'effects.svg');plt.close(fig)
    svg=ASSETS/'effects.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    lines += ['\n![固定人群与预选层/状态的效应](sink-next-20260925/effects.png)']
    if 4 in summaries:
        s=summaries[4];sink=s['groups']['sink'];normal=s['groups']['normal']
        lines += ['\n## 4. 自由生成','\n主终点：完整非空 boxed，且生成长度不足2048。不要与仅 boxed 或自动正确率混用。',
            '\n| 条件 | Sink 完整且未截断 /152 | Sink 正确 /152 | 正常完整且未截断 /50 | 正常正确 /50 | Sink 退出/进入（相对 zero） |\n|---|---:|---:|---:|---:|---|']
        for c,v in sink['arms'].items():
            n=normal['arms'][c]
            lines += [f"| {c} | {v['primary_count']} | {v['correct']} | {n['primary_count']} | {n['correct']} | {v['exit_vs_zero']} / {v['entry_vs_zero']} |"]
        for c,v in sink['primary_comparisons'].items():
            lines += [f"\nC1 对 {c}：改善{v['wins']}题，损伤{v['losses']}题，净{v['net']:+d}；双侧精确p={v['p_two_sided_exact']:.6g}，Holm校正p={v['holm_p']:.6g}。"]
        lines += [f"\n预定‘对两组均改善且显著’规则：**{s['decision']['improves_vs_both']}**。",'\n在线对照匹配本组当前激活上的更新规则，各组文本分叉后不保证累计能量相同。重新编码是在无干预模型上对新文本重放，不能等同于被干预时的在线轨迹。']
    else:lines += ['\n## 4. 自由生成','\n已冻结、按依赖顺序运行；完整审核结果尚未同步，当前不报告部分效应。152道sink题+50道正常题，zero/C1/ORTH_MAN_1，共606条记录。']
    lines += ['\n## 范围与追溯','\n第二个模型按用户规格留到讨论期。所有结果须在用户指定2026-09-26 01:59 UTC之前完成审核才能进入投稿版；这里不核实会议官方截止日期。',
        '\n| 实验 | 完成审核 UTC | 投稿冻结前 | Plan SHA256 |\n|---|---|---|---|']
    for i,s in summaries.items():lines += [f"| {i} | {s['completed_utc']} | {s['submission_eligible']} | `{s['plan_sha256']}` |"]
    lines += ['\n原始逐token logprobs、token IDs、更新量和收据在 Lambda `/lambda/nfs/dami/hss/sink-next-20260925/exp*/`；轻量副本在本地 `results/sink-next-20260925/`。完整10方向明细见各项 [JSON](sink-next-20260925/)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    archive=ROOT.parent/'sink-next-results-20260925.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in [REPORT,REPO/'docs/sink-next-protocol-20260925.zh-CN.md',*ASSETS.rglob('*')]:
            if p.is_file():z.write(p,p.relative_to(REPO/'docs'))
    print(json.dumps(dict(report=str(REPORT),experiments=list(summaries),zip_bytes=archive.stat().st_size)))


if __name__=='__main__':run()
