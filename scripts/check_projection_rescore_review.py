"""Compare automatic v1 scores with earlier sealed AI readings, without relabeling.

This is an in-sample diagnostic: those readings informed extraction rule design.
It is neither independent validation nor an exhaustive semantic review.
"""
import argparse
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path

from projection_rescore_v1 import canonical
from revision_common import sha, write_json


def read(p):return json.loads(p.read_text())


def run(root, primary):
    scored=root/'scoring_sensitivity_v1'; out=root/'scoring_sensitivity_v1_review_check'
    if out.exists():raise RuntimeError('Preserve existing diagnostic')
    out.mkdir()
    provenance=read(root/'semantic_review_v2/summary.json')
    known={}
    for r in provenance['rows']:
        for side,letter in [('left','A'),('right','B')]:
            path=r[side+'_record'];value=r['final_answer_correct_'+letter]
            assert path not in known or known[path]['correct']==value
            known[path]=dict(correct=value,note=r['note'],case_id=r['case_id'])
    rows=[json.loads(x) for x in (scored/'scores.jsonl').read_text().splitlines()]
    conflicts=[];integer_near=[];changed=read(scored/'changed_records.json')
    for r in rows:
        path=Path(r['path']);prefix=Path('/lambda/nfs/dami/hss')
        local=(root if path.relative_to(prefix).parts[0]==root.name else primary)/Path(*path.relative_to(prefix).parts[1:])
        assert sha(local)==r['source_sha256'];raw=read(local)
        if r['path'] in known and r['new_correct']!=known[r['path']]['correct']:
            conflicts.append(dict(r,prior_review=known[r['path']],reference=raw['ground_truth'],response=raw['response_text'],prompt=raw['prompt_text']))
        if r['new_correct']:
            try:
                a=Fraction(canonical(r['new_extracted'],r['task']));b=Fraction(canonical(raw['ground_truth'],r['task']))
                if a.denominator==b.denominator==1 and a!=b:
                    integer_near.append(dict(path=r['path'],sample_id=r['sample_id'],condition=r['condition'],candidate=str(a),reference=str(b)))
            except (ValueError,ZeroDivisionError):pass
    result=dict(complete=True,status='v1_not_accepted_as_clean_semantic_rescoring',
        scores_sha256=sha(scored/'scores.jsonl'),review_summary_sha256=sha(root/'semantic_review_v2/summary.json'),
        previously_reviewed_source_records=len(known),conflicts_with_prior_readings=len(conflicts),
        changed_labels=len(changed),changed_labels_in_prior_reading=sum(r['path'] in known for r in changed),
        changed_labels_not_previously_read=sum(r['path'] not in known for r in changed),
        conflicts=conflicts,unequal_integer_matches=integer_near,
        independent_validation=False,new_labels_adopted=False,
        next=['Preserve v1; general extraction fixes need a separate scorer version.',
              'Do not accept unequal integer final answers solely through relative tolerance.',
              'Use final-answer headings/terminal paragraph and predicate boundaries; avoid geometry names and Mr. as answers.',
              'Review all changed answers and unresolved extraction classes before claiming semantic correction.'],
        source_sha256=sha(__file__))
    write_json(out/'summary.json',result)
    e=lambda x:html.escape(str(x))
    body='<h1>评分 v1 复核：尚不能作为最终修正分数</h1>'
    body+='<div class="note">全量运行与统计复算已通过，但新提取规则仍有已知错误。此前已读过的244条源记录中，13条与已封存AI判断不一致。这是受已有案例影响的内部检查，不是独立验证。</div>'
    body+='<p>共有153个自动标签变化，其中23条源记录此前读过，130条尚未完成新的逐条语义核验。原始分数与实验选择均保留。</p>'
    body+='<p><a href="../scoring_sensitivity_v1/index.html">全量v1自动统计（非最终语义分数）</a> · <a href="../semantic_review_v2/index.html">原182对逐题核验</a> · <a href="summary.json">完整诊断记录</a></p>'
    body+='<h2>已经定位的问题</h2><ul><li>句子中的几何名称被误取为答案，如OBCD、ABCDEF；人名前的Mr.也被错误断句。</li><li>只寻找Therefore/Thus会错过后面的Final Answer或Conclusion章节。</li><li>继承的0.1%相对容差会把1005与1004判为相等；计数答案不应如此。</li><li>多个候选、进制下标、根式、答案后的限定语仍需更稳健的处理。</li></ul>'
    body+='<p>因此没有采用v1替换原标签，也没有据此重新选优。下一版须保留本版、修复通用规则并覆盖全部条件重算。</p>'
    body+='<h2>全部13条与原核验不一致的源记录</h2>'
    for i,r in enumerate(conflicts,1):
        body+='<details><summary>'+str(i)+'. '+e(r['sample_id']+' / '+r['condition'])+'：提取 '+e(r['new_extracted'])+'</summary><p>'+e(r['prior_review']['note'])+'</p><p>参考：'+e(r['reference'])+'</p><pre>'+e(r['prompt'])+'</pre><pre>'+e(r['response'])+'</pre></details>'
    body+='<h2>相对容差接受的不相等整数</h2><pre>'+e(json.dumps(integer_near,ensure_ascii=False,indent=2))+'</pre>'
    style='body{font:16px system-ui;max-width:1150px;line-height:1.7;margin:30px auto;padding:20px;color:#21343f}.note{background:#fff2df;padding:18px;border-left:4px solid #ae7331}details{border-bottom:1px solid #ddd;padding:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126d84}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>评分v1复核</title><style>'+style+'</style>'+body+'</html>')
    print(json.dumps({k:result[k] for k in ['status','previously_reviewed_source_records','conflicts_with_prior_readings','changed_labels_not_previously_read']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--primary-root',type=Path,required=True);a=p.parse_args();run(a.root,a.primary_root)
