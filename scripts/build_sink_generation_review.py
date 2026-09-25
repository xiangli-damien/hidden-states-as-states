"""Build a descriptive, explicitly incomplete answer review without rescoring the trial."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='results/sink-generation-review-20260925')
    args=parser.parse_args(); root=Path(args.root)
    snapshot=json.loads((root/'snapshot.json').read_text())
    notes=json.loads((root/'review_notes.json').read_text())
    assert hashlib.sha256((root/'snapshot.json').read_bytes()).hexdigest()==notes['snapshot_sha256']
    cohort={r['sample_id']:r for r in snapshot['cohort']}; grouped={}
    for item in snapshot['records']:
        r=item['record']; assert item['sha256']==item['receipt']['sha256']
        grouped.setdefault(r['question_id'],{})[r['condition']]=r
    reviewed={r['question_id']:r for r in notes['rows']}
    paired={q for q,v in grouped.items() if 'zero' in v and 'C1' in v}
    assert set(reviewed)==paired and len(paired)==43
    ids=sorted(paired)
    stats={}
    for name in ['all','sink','normal']:
        qq=[q for q in ids if name=='all' or grouped[q]['zero']['set']==name]
        arms={}
        for c in ['zero','C1']:
            arms[c]=dict(n=len(qq),boxed=sum(grouped[q][c]['primary_success'] for q in qq),
                frozen_auto_correct=sum(grouped[q][c]['correct'] for q in qq),
                reviewed_final_correct=sum(reviewed[q]['reviewed_final_correct_'+c] for q in qq),
                supported_solution=sum(reviewed[q][c+'_solution_supported'] for q in qq),
                mean_tokens=statistics.mean(grouped[q][c]['n_tokens'] for q in qq))
        stats[name]=arms
    gains=[q for q in ids if not reviewed[q]['reviewed_final_correct_zero'] and reviewed[q]['reviewed_final_correct_C1']]
    losses=[q for q in ids if reviewed[q]['reviewed_final_correct_zero'] and not reviewed[q]['reviewed_final_correct_C1']]
    box_gains=[q for q in ids if not grouped[q]['zero']['primary_success'] and grouped[q]['C1']['primary_success']]
    box_losses=[q for q in ids if grouped[q]['zero']['primary_success'] and not grouped[q]['C1']['primary_success']]
    missing=[c['question_id'] for c in snapshot['cases'] if c['question_id'] not in paired]
    summary=dict(snapshot_utc=snapshot['snapshot_utc'],source=snapshot['source'],records=len(snapshot['records']),
        pairs=len(ids),complete_triples=sum(len(grouped[q])==3 for q in ids),pending_pairs=len(missing),
        stats=stats,final_answer_gains=gains,final_answer_losses=losses,boxed_gains=box_gains,boxed_losses=box_losses,
        categories=dict(Counter(x['category'] for x in reviewed.values())),
        scope=notes['method'],snapshot_sha256=notes['snapshot_sha256'])
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    out=root/'report';out.mkdir(exist_ok=True)
    esc=html.escape
    rows=[]; sections=[]; md=[]
    label=lambda v:'对' if v else '错／未完成'
    for q in ids:
        n=reviewed[q];rs=grouped[q];z=rs['zero'];c=rs['C1']
        change='改善' if q in gains else '退步' if q in losses else '正误不变'
        if n['category']=='clean_repair':change='推理和答案修复'
        if n['category']=='answer_gain_invalid_proof':change='答案对了，推导仍错'
        if n['category']=='format_only':change='新增 boxed，数学仍错'
        if n['category']=='completion_regression':change='丢失 boxed／达到上限'
        rows.append(f'<tr data-key="{esc(q+" "+change+" "+n["note"])}"><td><a href="#{q}">{q}</a></td><td>{z["set"]}</td><td>{label(n["reviewed_final_correct_zero"])} → {label(n["reviewed_final_correct_C1"])}</td><td>{z["n_tokens"]} → {c["n_tokens"]}</td><td>{esc(change)}</td><td>{esc(n["note"])}</td></tr>')
        arms=[]
        for name in ['zero','C1','ORTH_MAN_1']:
            if name not in rs:
                arms.append('<section><h3>正交对照</h3><p>快照中尚未完成。</p></section>');continue
            r=rs[name];title={'zero':'不干预','C1':'sink 轴截断','ORTH_MAN_1':'正交对照（未逐题语义复核）'}[name]
            arms.append(f'<section><h3>{title}</h3><p>{r["n_tokens"]} tokens · boxed {r["boxed"]} · 冻结自动正确 {r["correct"]}</p><pre>{esc(r["response_text"])}</pre></section>')
        sections.append(f'<details id="{q}"><summary>{q} · {esc(change)}</summary><p>{esc(n["note"])}</p><h3>题目</h3><pre>{esc(cohort[q]["prompt_text"])}</pre><p>参考答案：{esc(str(cohort[q]["ground_truth"]))}</p><div class="arms">{"".join(arms)}</div></details>')
        md.append(f'| {q} | {z["set"]} | {label(n["reviewed_final_correct_zero"])}→{label(n["reviewed_final_correct_C1"])} | {z["n_tokens"]}→{c["n_tokens"]} | {n["note"]} |')
    intro=f'''# 截断后是否变好：43 题全文复核

快照：{snapshot['snapshot_utc']}。共 128 条已保存生成，43 道题有 zero/C1 配对（35 sink 候选题、8 normal），42 道也有对照。计划总计 202 题，另 159 题尚无配对可复核。快照固定，后来生成不混入本统计。

结论：没有普遍变好。3 题最终答案由错变对，2 题由对变错；3 个答案改善中只有 1 个得到有效推导，另外 2 个仍含关键推导错误。完整 boxed 的增加主要是完成形式变化。

这是可见组别和参考答案的 AI 辅助全文检查，不是独立人工盲审，也不是正式证明验证。原始自动标签、主要终点和统计协议都保留。控制组全文提供给读者，但本次未全部语义复核。重复的原样段落在阅读时折叠，报告原文完整保留。

## 三种口径分开

| 全部 43 题 | 不干预 | C1 |
|---|---:|---:|
| 完整 boxed 且未达长度上限 | 23 | 28 |
| 冻结自动判分正确 | 8 | 10 |
| 全文复核最终答案正确 | 11 | 12 |
| 本次复核认为推导足以支持答案（定性） | 11 | 10 |

最后一行只是本次定性检查：答案正确但存在关键推导错误不计入。不能拿这批中途结果做显著性结论，也不以此改实验参数。更短、更长或换了 token，都不自动算更好。

## 关键逐题发现

- **math_4508：明确修复。** 基线将 Re(1/z) 的分子 x 丢掉，得到 6π；C1 正确推导圆 (x−3)²+y²=9，答案 9π。
- **math_2508、math_2576：最终答案修复，推导未修复。** 分别答出 −46、3，但前者给四次多项式添加额外实根且系数运算错误，后者代换和 AM-GM 运用错误。
- **math_2786：真实损伤。** 原本完整正确地得到 d=4；截断后任意设 a=0，答成 1/c。
- **math_2734：自动分数漏掉的一次真实损伤。** 原本正确推得 b=1,2，只是未 boxed；截断后错误变成 b≤0 或 b≥4。
- **8 个新增 boxed 中，7 个答案仍错，1 个答案正确但推导仍错。** 另外 3 题丢失 boxed 并达到长度上限；这 3 题原答案也错。
- **math_2443、math_4460：重复减轻，但仍未解题。** 前者修正了展开的一个系数，却未得到所求多项式；后者结束循环但只给未定义的面积 A。
- **math_3670、math_79：两组均答对而被自动评分漏掉。** 原文分别明确回答 8 和 108，没有 boxed。

## “历史回答均值地图中的候选区域”具体含义

每道历史完整回答在第 14 层的生成 token 向量先取平均，形成一道题一个向量。冻结 GMM（K=33）划分这 5,000 个向量。训练部分中，选择人数≥50、完整 boxed 率≤30%且 boxed 率最低的簇，当前 local ID=8。这里不是逐 token 地图，也不是从 prompt 就断言新回答会进入该区域。

聚类本身不使用正确性标签；把哪个簇命名为候选，使用了训练回答的 boxed 信息，因此不能声称候选命名完全无行为信息。方向和阈值由 A/训练部分估计；原地图曾在所有 5,000 个无标签向量上拟合，是 transductive。当前候选是混合组成，未 boxed 可以是答案没格式化、不会求解或重复循环，不能统称死循环。

原稿曾举 global ID3 并描述 almost entirely empty or malformed；没有找到它与当前保存地图的对应证据，且当前收集没有空生成。因此不能宣称这是原论文 sink 的精确复现。这里的结果支持范围应称“当前 response-mean 地图的低 boxed 率候选区域”。

## 全部配对题目

| 题目 | 人群 | 复核最终答案 | token 数 | 复核说明 |
|---|---|---|---|---|
'''
    report=intro+'\n'.join(md)+'\n'
    (root/'review.zh-CN.md').write_text(report)
    doc=Path('docs/sink-generation-review-20260925.zh-CN.md');doc.write_text(report)
    title='截断后是否变好 · 43 题全文复核'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.6 system-ui;margin:32px;color:#172033;background:#fafbfc}}h1{{font-size:28px}}.notice{{padding:18px;background:#fff2d5;border-left:4px solid #ab7300}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce1e9;vertical-align:top}}th{{position:sticky;top:0;background:#eef2f6}}details{{margin:18px 0;border:1px solid #dce1e9;background:white;padding:16px;scroll-margin-top:25px}}summary{{cursor:pointer;font-weight:700}}.arms{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace}}input{{padding:12px;width:min(650px,90%);margin:20px 0}}.cards{{display:flex;gap:24px;flex-wrap:wrap}}.cards p{{background:#eef2f6;padding:18px}}@media(max-width:1000px){{.arms{{grid-template-columns:1fr}}}}</style>
<h1>{title}</h1><p>固定快照 {esc(snapshot['snapshot_utc'])} · 35 道候选 sink + 8 道 normal · 已读完全部 43 对 zero/C1 · 159 题尚待生成与复核</p>
<div class="notice"><b>没有普遍变好。</b>3 题最终答案改善、2 题退步；3 个改善中只有 1 个推导明确修复。新增 boxed 多数仍答错。AI 辅助非盲审，原始评分不改；对照全文可查看，本次未全部语义复核。尚无全量显著性结论。</div>
<div class="cards"><p>完整 boxed<br><b>23 → 28</b></p><p>冻结自动正确<br><b>8 → 10</b></p><p>全文最终答案正确<br><b>11 → 12</b></p><p>定性有效解答<br><b>11 → 10</b></p></div>
<p><a href="../review.zh-CN.md">完整说明 Markdown</a> · <a href="../summary.json">可复算汇总</a> · <a href="../review_notes.json">逐题复核 JSON</a></p>
<input id="search" placeholder="按题号、修复、退步、boxed 或错误原因筛选表格"><table><thead><tr><th>题号</th><th>人群</th><th>最终答案</th><th>tokens</th><th>变化</th><th>依据</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>原题、参考答案与完整三组原文</h2><p>点击题号或展开下方卡片。原文保留 LaTeX，未删改。</p>{''.join(sections)}
<h2>尚待配对复核的 159 题</h2><p>{esc(', '.join(missing))}</p>
<script>document.querySelector('#search').addEventListener('input',e=>{{let s=e.target.value.toLowerCase();document.querySelectorAll('tr[data-key]').forEach(r=>r.hidden=!r.dataset.key.toLowerCase().includes(s));}});function reveal(){{let x=document.getElementById(location.hash.slice(1));if(x?.tagName==='DETAILS')x.open=true;}}window.addEventListener('hashchange',reveal);reveal();</script></html>'''
    (out/'index.html').write_text(body)
    for q in ids:
        assert body.count(f'id="{q}"')==1
    assert all(esc(grouped[q][c]['response_text']) in body for q in ids for c in grouped[q])
    for target in ['review.zh-CN.md','summary.json','review_notes.json']:
        assert (root/target).exists()
    (root/'review_build_audit.json').write_text(json.dumps(dict(passed=True,full_zero_C1_pairs=43,
        original_texts_embedded=len(snapshot['records']),pending_pairs=len(missing),
        report_sha256=hashlib.sha256((out/'index.html').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
