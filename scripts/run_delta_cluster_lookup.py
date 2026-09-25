"""Direct, separate-layer empirical error lookup on frozen token-delta GMMs."""
import argparse
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from delta_cluster_lookup_common import token_counts, fit_error_lookup, lookup_sequence
from hss.analysis.change_bayes import bootstrap_auc
from revision_common import config, freeze, provenance, sha, write_json, write_npz


def load_data(cfg):
    src = Path(cfg['source'])
    rows = pd.read_parquet(src / 'rows.parquet')
    splits = pd.read_parquet(src / 'splits.parquet')
    np.testing.assert_array_equal(rows.sample_id, splits.sample_id)
    assert rows.sample_id.is_unique and rows.question_group.is_unique and len(rows) == 5000
    rows['split'] = splits.split
    protocol = json.loads((src / 'protocol.json').read_text())
    assert protocol['config']['layers'] == cfg['layers'] and protocol['normalization'] is False
    files = [src / f for f in ['rows.parquet', 'splits.parquet', 'protocol.json', 'PREPARED.json', 'ASSIGNED.json']]
    cards = []
    for layer in cfg['layers']:
        folder = src / 'fits' / f'token_delta_L{layer:02d}'
        selection = json.loads((folder / 'selection.json').read_text())
        cards.append(selection['k'])
        files += [folder / 'selection.json', folder / 'selected.joblib']
    assert cards == [32] * 5
    sequences = [None] * len(rows)
    shards = []
    for record in json.loads((src / 'PREPARED.json').read_text())['shards']:
        p = src / 'assignments' / (record['shard'] + '.npz')
        receipt = p.with_suffix('.json')
        assert sha(p) == json.loads(receipt.read_text())['sha256']
        files += [p, receipt]
        with np.load(p) as z:
            ptr, qi, sequence = z['token_ptr'], z['question_index'], z['selected_token_delta']
            assert ptr[0] == 0 and ptr[-1] == len(sequence) and (np.diff(ptr) > 0).all()
            for j, i in enumerate(qi):
                assert sequences[i] is None
                sequences[i] = sequence[ptr[j]:ptr[j+1]].copy()
            shards.append({'name': record['shard'], 'question_index': qi.copy(), 'token_ptr': ptr.copy()})
    assert all(s is not None for s in sequences)
    np.testing.assert_array_equal([len(s) for s in sequences], rows.n_tokens)
    return rows, sequences, shards, cards, files


def count_table(rows, counts, layers, q):
    result = []
    for j, layer in enumerate(layers):
        for k in range(counts.shape[2]):
            rec = {'layer': layer, 'cluster': k, 'lookup_error_rate': float(q[j, k])}
            for split in ['train', 'validation', 'test']:
                chosen = rows.split.eq(split).to_numpy()
                v = counts[chosen, j, k]
                y = 1 - rows.label.to_numpy(int)[chosen]
                wrong, correct = int(v[y == 1].sum()), int(v[y == 0].sum())
                rec.update({split + '_wrong_tokens': wrong, split + '_correct_tokens': correct,
                            split + '_tokens': wrong + correct, split + '_questions': int((v > 0).sum()),
                            split + '_error_rate': wrong / (wrong + correct) if wrong + correct else None})
            result.append(rec)
    return pd.DataFrame(result)


def independent_auc(y, scores):
    wrong = scores[y == 1]
    correct = np.sort(scores[y == 0])
    return float((np.searchsorted(correct, wrong, side='left').sum()
                  + np.searchsorted(correct, wrong, side='right').sum()) / (2 * len(wrong) * len(correct)))


def run(cfg, cfg_path):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / '_SUCCESS.json').exists():
        raise RuntimeError('Already complete; do not overwrite frozen experiment')
    if (root / 'plan.json').exists():
        raise RuntimeError('Partial run exists; preserve it and use an explicit recovery namespace')
    start = time.time()
    def progress(stage, **info):
        write_json(root / 'status.json', {'state': 'running', 'phase': stage, 'pid': os.getpid(),
                   'started_unix': start, 'updated_unix': time.time(), **info})
        print(json.dumps({'stage': stage, **info}), flush=True)
    progress('load')
    rows, sequences, shards, cards, files = load_data(cfg)
    lengths = rows.n_tokens.to_numpy(int); y = 1 - rows.label.to_numpy(int)
    train = rows.split.eq('train').to_numpy()
    chosen = sorted(rows.loc[rows.split.eq('test'), 'sample_id'],
                    key=lambda s: hashlib.sha256(f'{cfg["seed"]}:{s}'.encode()).hexdigest())[:cfg['case_count']]
    files += [Path(cfg_path).resolve(), Path(__file__).resolve(), Path(__file__).with_name('delta_cluster_lookup_common.py'),
              Path(__file__).with_name('revision_common.py'),
              Path(__file__).resolve().parents[1] / 'src/hss/analysis/change_bayes.py',
              Path(cfg['previous']) / 'tokens.json', Path(cfg['tokenizer'])]
    plan = provenance(cfg, files)
    plan.update(selected_case_ids=chosen, k_per_layer=cards, training_questions=int(train.sum()),
                q_definition='Raw training wrong-token count / training total-token count. No smoothing or question weighting.',
                score_definition='Current token lookup; separately within-layer arithmetic mean of observed lookups. No layer fusion.',
                fallback='Unseen training cluster uses training token-label prior; explicitly flagged.')
    freeze(root / 'plan.json', plan)
    progress('training_lookup')
    counts = np.stack([token_counts(s, cards) for s in sequences])
    lookup = fit_error_lookup(counts, y, train)
    q = lookup['q']
    write_npz(root / 'lookup.npz', layers=np.asarray(cfg['layers']), **lookup)
    table = count_table(rows, counts, cfg['layers'], q)
    # The persisted q comes only from train; held-out counts never feed it.
    table.to_csv(root / 'cluster_rates.csv', index=False)
    maps = root / 'maps'; maps.mkdir()
    for layer in cfg['layers']:
        shutil.copyfile(Path(cfg['source']) / 'fits' / f'token_delta_L{layer:02d}' / 'selected.joblib', maps / f'L{layer:02d}.joblib')
    freeze(root / 'LOOKUP_FROZEN.json', {'lookup_sha256': sha(root / 'lookup.npz'), 'plan_sha256': sha(root / 'plan.json'),
           'training_token_count': int(lengths[train].sum()), 'token_error_prior': lookup['token_error_prior'].tolist(),
           'empty_training_clusters': np.argwhere(lookup['fallback']).tolist(),
           'maps': {p.name: sha(p) for p in maps.iterdir()}})
    progress('token_scores')
    token_dir = root / 'token_scores'; token_dir.mkdir()
    snapshots = []; audits = {'question_counts_checked': 0, 'token_layer_lookups_checked': 0, 'prefix_scores_checked': 0}
    reference_counts = np.zeros_like(counts)
    all_current = {}; all_mean = {}
    # All generated tokens get a score, but only visible prefixes enter online evaluation.
    for shard in shards:
        qi, ptr = shard['question_index'], shard['token_ptr']
        raw_list = []; mean_list = []
        for i in qi:
            s = sequences[i]
            raw, mean = lookup_sequence(s, q)
            raw_list.append(raw); mean_list.append(mean)
            for j in range(len(cards)):
                reference_counts[i, j] = np.histogram(s[:, j], bins=np.arange(cards[j] + 1) - .5)[0]
                np.testing.assert_array_equal(raw[:, j], q[j, s[:, j]])
                np.testing.assert_allclose(mean[:, j] * np.arange(1, len(s) + 1), np.add.accumulate(q[j, s[:, j]]), atol=1e-10)
            audits['question_counts_checked'] += 1
            audits['token_layer_lookups_checked'] += int(raw.size)
            for p in cfg['prefixes']:
                if len(s) <= p: continue  # answer still ongoing after p observed response tokens
                for j, layer in enumerate(cfg['layers']):
                    snapshots.append({'sample_id': rows.sample_id.iloc[i], 'split': rows.split.iloc[i], 'failure': int(y[i]),
                                      'prefix': p, 'layer': layer, 'current_token': float(raw[p-1, j]),
                                      'prefix_mean': float(mean[p-1, j])})
                    ref = sum(float(q[j, int(k)]) for k in s[:p, j]) / p
                    assert abs(ref - mean[p-1, j]) < 1e-12
                    audits['prefix_scores_checked'] += 2
            if rows.sample_id.iloc[i] in chosen:
                all_current[rows.sample_id.iloc[i]] = raw
                all_mean[rows.sample_id.iloc[i]] = mean
        write_npz(token_dir / (shard['name'] + '.npz'), question_index=qi, token_ptr=ptr,
                  cluster_ids=np.concatenate([sequences[i] for i in qi]),
                  current_token=np.concatenate(raw_list), prefix_mean=np.concatenate(mean_list))
    np.testing.assert_array_equal(counts, reference_counts)
    for j in range(len(cards)):
        for k in range(cards[j]):
            a = int(reference_counts[train & (y == 1), j, k].sum())
            b = int(reference_counts[train & (y == 0), j, k].sum())
            if a + b: assert q[j, k] == a / (a + b)
    frame = pd.DataFrame(snapshots).sort_values(['prefix', 'layer', 'sample_id'])
    frame.to_parquet(root / 'prefix_scores.parquet', index=False)
    freeze(root / 'SCORES_FROZEN.json', {'lookup_sha256': sha(root / 'lookup.npz'),
           'scores_sha256': sha(root / 'prefix_scores.parquet'),
           'token_score_files': {p.name: sha(p) for p in token_dir.iterdir()}})
    progress('test_metrics')
    records = []; coverage = []
    for p in cfg['prefixes']:
        cohort = frame[(frame.prefix == p) & (frame.split == 'test')]
        ids = sorted(cohort.sample_id.unique())
        yy = cohort[cohort.layer.eq(cfg['layers'][0])].set_index('sample_id').loc[ids, 'failure'].to_numpy()
        rng = np.random.default_rng(cfg['seed'] + p)
        boot = np.column_stack([rng.choice(np.flatnonzero(yy == c), (cfg['bootstrap'], int((yy == c).sum())), replace=True) for c in [1, 0]])
        write_npz(root / f'bootstrap_p{p}.npz', sample_ids=np.asarray(ids), indices=boot)
        coverage.append({'prefix': p, 'test_questions': len(ids), 'test_wrong': int(yy.sum()),
                         'all_eligible': int((lengths > p).sum())})
        for layer in cfg['layers']:
            layer_frame = cohort[cohort.layer.eq(layer)].set_index('sample_id').loc[ids]
            for mode in ['current_token', 'prefix_mean']:
                score = layer_frame[mode].to_numpy()
                auc = float(roc_auc_score(yy, score)); draws = bootstrap_auc(yy, score, boot)
                ref_draws = np.asarray([independent_auc(yy[ix], score[ix]) for ix in boot])
                np.testing.assert_allclose(draws, ref_draws, atol=1e-12)
                assert abs(auc - independent_auc(yy, score)) < 1e-12
                rec = {'prefix': p, 'layer': layer, 'mode': mode, 'test_questions': len(ids), 'auroc': auc,
                       'ci_low': float(np.quantile(draws, .025)), 'ci_high': float(np.quantile(draws, .975))}
                records.append(rec)
        progress('test_metrics', completed_prefix=p)
    metrics = pd.DataFrame(records); metrics.to_csv(root / 'metrics.csv', index=False)
    audits.update(complete=True, metrics_checked=len(records), independent_bootstrap_auc_checks=len(records)*cfg['bootstrap'])
    write_json(root / 'audit.json', audits)
    progress('report')
    report(cfg, rows, sequences, chosen, all_current, all_mean, table, metrics, coverage)
    for f, h in plan['files'].items(): assert sha(f) == h, f
    write_json(root / '_SUCCESS.json', {'seconds': time.time()-start, 'questions': len(rows), 'tokens': int(lengths.sum()),
               'training_tokens': int(lengths[train].sum()), 'layers': cfg['layers'], 'metrics': len(records),
               'summary_sha256': sha(root / 'summary.json'), 'lookup_sha256': sha(root / 'lookup.npz')})
    write_json(root / 'status.json', {'state': 'complete', 'phase': 'report', 'pid': os.getpid(), 'started_unix': start,
               'finished_unix': time.time(), 'delivery': 'visual_review_pending'})


def report(cfg, rows, sequences, chosen, all_current, all_mean, table, metrics, coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(cfg['output']); out = root / 'report'; out.mkdir()
    layers = cfg['layers']; labels = [f'{l-1}→{l}' for l in layers]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained', sharey=True)
    for ax, mode, title in zip(axes, ['current_token', 'prefix_mean'], ['Current token lookup', 'Mean of observed token lookups (one layer only)']):
        for layer, lab in zip(layers, labels):
            sub = metrics[(metrics.layer == layer) & (metrics['mode'] == mode)].sort_values('prefix')
            ax.plot(sub.prefix, sub.auroc, marker='o', label=lab)
        ax.axhline(.5, color='gray', linestyle='--'); ax.set(xlabel='Observed generated tokens', ylabel='Failure AUROC', title=title); ax.grid(alpha=.15); ax.legend(title='Layer pair')
    fig.savefig(out / 'layer_auc.png', dpi=160); fig.savefig(out / 'layer_auc.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3.5), layout='constrained')
    mat = np.stack([table[table.layer.eq(l)].sort_values('cluster').lookup_error_rate for l in layers])
    im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='coolwarm'); ax.set_yticks(range(5), labels)
    ax.set(xlabel='Layer-local cluster ID (IDs are NOT matched across rows)', title='Training-token error proportion per cluster')
    fig.colorbar(im, ax=ax, label='Fraction from incorrect training answers'); fig.savefig(out/'cluster_rates.png', dpi=160); plt.close(fig)
    token_meta = json.loads((Path(cfg['previous']) / 'tokens.json').read_text())
    ids = {sid: token_meta[sid]['response_ids'] for sid in chosen}
    # Tokenizer-only helper uses an already-installed environment; never loads a model/GPU.
    code = 'import sys,json;from tokenizers import Tokenizer;t=Tokenizer.from_file(sys.argv[1]);d=json.load(sys.stdin);json.dump({k:[t.decode([x],skip_special_tokens=False) for x in v] for k,v in d.items()},sys.stdout)'
    pieces = json.loads(subprocess.check_output([cfg['tokenizer_python'], '-c', code, cfg['tokenizer']], input=json.dumps(ids), text=True))
    css = '<style>body{max-width:1120px;margin:28px auto;padding:0 22px;font:16px/1.7 system-ui;color:#23354a}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #cad3dd;padding:6px 9px}th{background:#edf3f8}img{max-width:100%}.scroll{overflow:auto}.note{background:#fff3d6;padding:16px}pre{white-space:pre-wrap}input{width:90%}</style>'
    wrap = lambda title, body: '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title>'+css+'<body>'+body+'</body></html>'
    body = '<h1>逐层 Δh 簇错误率：直接查表</h1><p class="note">每层独立。没有跨层合并、贝叶斯路线分类器、转移评分或熵融合。q = 该簇来自错误训练回答的 token 数 ÷ 该簇全部训练 token 数。按原始 token 计数，无平滑、无题目等权。标签是整题最终对错，不是这个 token 的步骤对错。</p>'
    body += '<h2>怎么看分数</h2><p><b>当前 token：</b>用它在这一层的 Δh 找到簇，直接读取 q。<b>截至当前的平均：</b>只对同一层已经出现的 q 求简单平均；仍然每层各有一个分数。后者是排序分数，不是已校准的整题错误概率。所有测试分数仅依赖已出现的前缀。</p><p>复用五个冻结 GMM，每层 K32。旧 GMM 用每道训练题抽4个 token 拟合；本次 q 使用训练3011题的全部生成 token。没有重新拟合 GMM，也没有覆盖所有28层。训练/验证/测试按题分开。</p>'
    body += '<h2>五层分别的测试结果</h2><img src="layer_auc.png"><p>每个时间点只评估仍在生成的回答。表中 AUROC 在测试问题之间计算，不把同题 token 当成独立测试样本。95%区间按题 bootstrap，未做多重比较校正；数据与自动标签均来自已探索的历史实验。</p><h3>主时间点：已生成64个 token</h3><div class="scroll">'+metrics[metrics.prefix.eq(cfg['primary_prefix'])].to_html(index=False,float_format=lambda v:f'{v:.4f}')+'</div>'
    body += '<h2>每簇的训练错误率</h2><img src="cluster_rates.png"><p>每行是独立地图，跨行相同编号不代表同一簇。点击层对查看训练、验证和测试的原始计数；测试比例不会回写训练查表。</p><ul>'
    for layer, lab in zip(layers, labels):
        name=f'layer_{layer:02d}.html'; body += f'<li><a href="{name}">{lab}：32个簇的计数与错误比例</a></li>'
        sub=table[table.layer.eq(layer)]
        (out/name).write_text(wrap('层对 '+lab, '<h1>层对 '+lab+'</h1><a href="index.html">返回</a><p>lookup_error_rate来自训练token；correct/wrong均指来源回答的最终自动评分。验证/测试列仅作外部检查。</p><div class="scroll">'+sub.to_html(index=False,float_format=lambda v:f'{v:.4f}')+'</div>'))
    body+='</ul><h2>逐题、逐token查看</h2><p>预先按哈希选32道历史测试题，不按新结果筛选。热图显示前128个token；滑块可查看整段回答的每个位置。单token解码可能显示字符碎片，原回答保留在下方。</p><ul>'
    index = rows.set_index('sample_id')
    for sid in chosen:
        row = index.loc[sid]; i = int(token_meta[sid]['row_index']); assert rows.sample_id.iloc[i] == sid
        raw, mean = all_current[sid], all_mean[sid]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), layout='constrained')
        n = min(128, len(raw)); im=axes[0].imshow(raw[:n].T,aspect='auto',vmin=0,vmax=1,cmap='coolwarm',extent=[.5,n+.5,4.5,-.5]); axes[0].set_yticks(range(5),labels);axes[0].set(title=sid+' — current token lookup',xlabel='Observed token position');fig.colorbar(im,ax=axes[0])
        for j, lab in enumerate(labels): axes[1].plot(np.arange(1,n+1),mean[:n,j],label=lab)
        axes[1].set(xlabel='Observed token position',ylabel='Prefix average',ylim=(0,1));axes[1].legend(ncol=5)
        fig.savefig(out/(sid+'.png'),dpi=130);plt.close(fig)
        data={'sample_id':sid,'layers':layers,'token_ids':ids[sid],'token_text':pieces[sid], 'cluster_ids':sequences[i].tolist(), 'current_token':raw.tolist(),'prefix_mean':mean.tolist()}
        write_json(out/(sid+'.json'),data)
        payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
        case='<h1>'+sid+'</h1><a href="index.html">返回</a><p>原有整题自动评分：'+('正确' if row.label else '错误')+'；颜色不是步骤错误标签。</p><h2>问题</h2><pre>'+html.escape(row.prompt_text)+'</pre><img src="'+sid+'.png"><h2>逐token查表</h2><input id="pos" type="range" min="1" max="'+str(len(raw))+'" value="1"><p id="selected"></p><table><thead><tr><th>层对</th><th>簇ID</th><th>当前token错误率q</th><th>截至当前平均</th></tr></thead><tbody id="values"></tbody></table><h2>原回答</h2><pre>'+html.escape(row.response_text)+'</pre>'
        js='<script>const d='+payload+';const slider=document.getElementById("pos");function show(){const i=Number(slider.value)-1;document.getElementById("selected").textContent="位置 "+(i+1)+" / "+d.token_ids.length+"；token ID "+d.token_ids[i]+"；文本 "+JSON.stringify(d.token_text[i]);document.getElementById("values").innerHTML=d.layers.map((l,j)=>"<tr><td>"+(l-1)+"→"+l+"</td><td>"+d.cluster_ids[i][j]+"</td><td>"+d.current_token[i][j].toFixed(4)+"</td><td>"+d.prefix_mean[i][j].toFixed(4)+"</td></tr>").join("");}slider.addEventListener("input",show);show();</script>'
        (out/(sid+'.html')).write_text(wrap(sid,case+js));body+='<li><a href="'+sid+'.html">'+sid+'</a></li>'
    body+='</ul><h2>各时间点覆盖</h2>'+pd.DataFrame(coverage).to_html(index=False)+'<h2>全部逐层指标</h2><div class="scroll">'+metrics.to_html(index=False,float_format=lambda v:f'{v:.4f}')+'</div><p>这是保存数据上的前缀回放，尚未接入实时generate循环。本轮没有自由生成或steering，也没有重新选择K、增加层或融合特征。</p>'
    (out/'index.html').write_text(wrap('逐层 Δh 簇错误率',body))
    write_json(root/'summary.json',{'definition':'Raw training-token wrong/(wrong+correct), each layer separate.',
               'layers':layers,'coverage':coverage,'metrics':metrics.to_dict('records'),
               'primary_prefix':cfg['primary_prefix'],'cases':chosen,
               'limits':['Historical automated answer labels; not step-error labels.','Previously explored test questions.',
                         'No cross-layer score, no entropy, no transition scoring.','Pointwise question-bootstrap intervals; no multiple-comparison correction.',
                         'GMM reused from 12044 sampled training tokens, lookup estimated from all training tokens.',
                         'Offline causal-prefix replay; not new live generation or steering.']})
    write_json(root/'report_SUCCESS.json',{'summary_sha256':sha(root/'summary.json'),'files':{p.name:sha(p) for p in out.iterdir()}})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=config(args.config)
    try:
        with threadpool_limits(cfg['threads']): run(cfg,args.config)
    except Exception as exc:
        root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
        if not (root/'_SUCCESS.json').exists():
            write_json(root/'failure.json',{'error':repr(exc),'time':time.time(),'pid':os.getpid()})
        raise
