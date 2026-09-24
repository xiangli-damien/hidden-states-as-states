"""Verify token-control outputs and publish the complete diagnostic table."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from revision_common import paired_ratio_ci,write_json


def run(root):
    plan=json.loads((root/'plan.json').read_text())
    expected={f'p{p}_l{l}_{v}' for p,l,v in plan['config']['views']}
    assert {p.parent.name for p in root.glob('p*/_SUCCESS.json')}==expected
    rows=[]
    for name in sorted(expected):
        directory=root/name;d=json.loads((directory/'summary.json').read_text())
        marker=json.loads((directory/'_SUCCESS.json').read_text())
        for key,filename in [('summary_sha256','summary.json'),('per_question_sha256','per_question.parquet')]:
            assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==marker[key]
        frame=pd.read_parquet(directory/'per_question.parquet')
        assert len(frame)==d['test_questions'] and not frame.sample_id.duplicated().any()
        assert frame.split.eq('test').all()
        for method in d['methods']:
            ci=paired_ratio_ci(frame[method['method']+'_error'],frame.train_centered_energy)
            delta=paired_ratio_ci(frame[method['method']+'_error']-frame.gmm_centroid_error,frame.train_centered_energy)
            for key in ('estimate','ci95'):
                np.testing.assert_allclose(ci[key],method['nmse'][key],rtol=0,atol=1e-12)
                np.testing.assert_allclose(delta[key],method['nmse_minus_GMM'][key],rtol=0,atol=1e-12)
        methods={m['method']:m for m in d['methods']}
        rows.append({'view':name,'test_questions':d['test_questions'],'distinct_token_windows':d['test_distinct_token_windows'],
            'position_NMSE':methods['position_mean']['nmse']['estimate'],
            'token_ID_NMSE':methods['token_identity_mean']['nmse']['estimate'],
            'GMM_NMSE':d['GMM_centroid_nmse']['estimate'],'train_vocabulary':d['train_token_vocabulary'],
            'token_ID_coverage':d['test_known_token_fraction'],'GMM_K':d['GMM_K']})
    audit={'views':len(rows),'summary_prediction_hashes_and_bootstrap_valid':True,'table':rows}
    write_json(root/'audit.json',audit)
    dest=root/'report';dest.mkdir(exist_ok=True)
    (dest/'index.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Token geometry controls</title>
<style>body{font:16px system-ui;margin:35px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}</style>
<h1>Token 身份／位置能解释多少重构效果？</h1>
<p>训练集均值；原始 pre-norm 表示；问题级固定划分。NMSE 越低越好，1 为训练整体均值基线。</p>
'''+pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.6f}')+'''
<ul><li>每列比较在同一个 view 的相同测试题与同一分母上进行。不同 view 的分母及可用题目不同，不是控制实验。</li>
<li>chat prompt 尾部测试窗口的 token 序列完全相同，但其 hidden states 仍依赖前面的题目。低误差不能证明只编码模板。</li>
<li>位置基线有 16 个中心；GMM 有 64 个中心。token-ID 基线的中心数等于训练词表大小；正文/生成词表远大于64，不是同容量方法比较。</li>
<li>未知 token 用训练整体均值恢复。没有使用正确性标签，也没有测试集调参。</li>
<li>大部分总体中心化能量可由固定位置解释，不代表剩余能量没有功能。需要实际干预判断。</li>
<li>本实验是 MATH 历史测试集上的探索性诊断，不是 steering 或跨数据集结果。</li></ul></html>''')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    run(p.parse_args().root)
