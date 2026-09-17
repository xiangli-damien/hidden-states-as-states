"""Portable HTML review: real trial figures, metrics and escaped original cases.

Raw OpenAct text is exported once. Subsequent rendering never fits a model. The
case viewer works from file:// (external JS data, no fetch/server dependency).
"""

import html
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..analysis.quality import cluster_quality, prediction_quality, state_outcomes
from ..analysis.tables import (
    bootstrap_predictions,
    collect_tables,
    compare_trials,
    dynamics,
    selection_surface,
    state_map,
    trajectory_similarity,
)
from ..data import prepare
from ..experiments.artifacts import digest, file_digest, save_json
from ..experiments.config import load
from ..provenance import stage_version
from ..results import Result, ResultCatalog
from ..results.models import load_layer
from . import plots
from .artifacts import FigureBundle


def javascript(value):
    """Serialize as data; even closing script tags from a response stay inert."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def declared_results(root, state):
    """Find completed declared trials even when a controller has not registered them.

    Match the scientific config and recompute the exact identity from its saved
    snapshot. This lookup never prepares features or fits a model.
    """
    from ..experiments.runner import trial_identity

    results = {
        name: Result(j["path"])
        for name, j in state["jobs"].items()
        if j["status"] == "complete"
    }
    catalogs = {}
    for path in sorted((root / "configs").glob("*.json")):
        if path.stem in results:
            continue
        cfg = load(path)
        folder = cfg.execution.output_root
        if folder not in catalogs:
            catalogs[folder] = ResultCatalog(folder).results
        matches = []
        for result in catalogs[folder]:
            if any(
                result.config[k] != cfg.to_dict()[k]
                for k in (
                    "data",
                    "seed",
                    "cluster",
                    "transform",
                    "alignment",
                    "evaluation",
                )
            ):
                continue
            version = state.get("source_version", result.summary["source_version"])
            expected = digest(
                trial_identity(cfg, SimpleNamespace(info=result.snapshot), version)
            )
            if expected == result.summary["trial_id"]:
                matches.append(result)
        if len(matches) > 1:
            raise ValueError(
                f"Multiple completed snapshots match {path}; resolve the study ledger explicitly"
            )
        if matches:
            results[path.stem] = matches[0]
    return results


def export_cases(data, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "cases-manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["snapshot"] == data.info["key"] and all(
            (target / p).is_file() and file_digest(target / p) == h
            for p, h in saved["files"].items()
        ):
            return saved
    frames = []
    for source in data.info["identity"]["sources"]:
        root = Path(source["path"])
        for name in ("data.parquet", "labels/correctness.parquet"):
            if (
                name in source["files"]
                and file_digest(root / name) != source["files"][name]
            ):
                raise ValueError(f"Source changed since snapshot: {root / name}")
        raw = pd.read_parquet(root / "data.parquet")
        wanted = [
            "sample_id",
            "sample_idx",
            "problem",
            "prompt_text",
            "model_input_text",
            "response_text",
            "ground_truth",
            "solution",
            "finish_reason",
            "category",
            "level",
            "n_response_tokens",
        ]
        raw = raw[[k for k in wanted if k in raw]].copy()
        labels_path = root / "labels/correctness.parquet"
        if labels_path.exists():
            labels = pd.read_parquet(labels_path)
            keep = [
                c
                for c in (
                    "sample_idx",
                    "is_correct",
                    "extracted_answer",
                    "normalized_answer",
                    "error",
                    "meta_parse_failed",
                    "meta_gt_missing",
                )
                if c in labels
            ]
            raw = raw.merge(
                labels[keep], on="sample_idx", how="left", validate="one_to_one"
            )
        frames.append(raw)
    frame = pd.concat(frames, ignore_index=True)
    frame = (
        data.meta[["sample_id"]]
        .drop_duplicates()
        .merge(frame, on="sample_id", validate="one_to_one")
    )
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    (target / "cases.js").write_text("window.HSS_CASES=" + javascript(records) + ";\n")
    frame.to_parquet(target / "original_cases.parquet", index=False)
    accuracy = pd.to_numeric(frame.get("is_correct"), errors="coerce")
    summary = dict(
        snapshot=data.info["key"],
        n=len(frame),
        correct=int(accuracy.eq(1).sum()),
        incorrect=int(accuracy.eq(0).sum()),
        missing_labels=int(accuracy.isna().sum()),
        truncated=int(frame.finish_reason.eq("length").sum()),
        total_tokens=int(frame.n_response_tokens.sum())
        if "n_response_tokens" in frame
        else None,
        parse_failed=int(frame.meta_parse_failed.fillna(False).sum())
        if "meta_parse_failed" in frame
        else None,
        files={
            n: file_digest(target / n) for n in ("cases.js", "original_cases.parquet")
        },
    )
    save_json(marker, summary)
    return summary


CSS = """
body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#172638;margin:0}
main{max-width:1320px;margin:auto;padding:30px}h1{font-size:32px}h2{margin-top:38px}
a{color:#165f9c}header,.card,details{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:20px;margin:14px 0}
.muted{color:#586b7d}.metrics{display:flex;gap:14px;flex-wrap:wrap}.metric{padding:14px 22px;background:#e8f1f8;border-radius:8px}
table{border-collapse:collapse;max-width:100%;font-size:13px}th,td{padding:7px 10px;border-bottom:1px solid #dce3ea;text-align:left}th{white-space:nowrap}
.scroll{overflow:auto}input,select,button{font:inherit;padding:8px;border:1px solid #b9c8d5;border-radius:6px;background:white;margin:4px}
button{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f5f7fa;padding:16px;border-radius:6px}
img{width:100%;height:auto}summary{cursor:pointer;font-weight:600}.tag{display:inline-block;font-size:12px;background:#e8f1f8;border-radius:5px;padding:5px;margin:3px}
.warn{border-left:4px solid #b77624;padding-left:16px}.small{font-size:13px}
"""


VIEWER = r"""
<section id="cases" class="card"><h2>逐题查看 / Original questions & responses</h2>
<p class="muted">状态编号仅在同一个 trial 内可比较。生成均值轨迹是回答生成后对各层的汇总，不是逐 token 的时间轨迹。</p>
<input id="query" placeholder="搜索题目、回答或 sample ID" style="min-width:320px"><select id="correct"><option value="">全部正确性</option><option value="true">正确</option><option value="false">错误</option></select>
<select id="category"><option value="">全部类别</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span>
<div><select id="mapFilter" aria-label="聚类方法"><option value="">不按聚类状态筛选</option></select><select id="layerFilter" aria-label="层"><option value="">任意层</option></select><select id="stateFilter" aria-label="状态"><option value="">任意状态</option></select></div>
<div class="scroll"><table><thead><tr><th>ID</th><th>类别 / Level</th><th>正确</th><th>题目</th></tr></thead><tbody id="list"></tbody></table></div>
<div id="detail"></div></section><script src="cases.js"></script><script src="trajectories.js"></script>
<script>
const cases=window.HSS_CASES||[], maps=window.HSS_TRAJECTORIES||{};
const el=id=>document.getElementById(id);let page=0,filtered=cases;
const text=(tag,value,parent)=>{const x=document.createElement(tag);x.textContent=value??'';parent.appendChild(x);return x;};
for(const cat of [...new Set(cases.map(c=>c.category).filter(Boolean))].sort()){const o=document.createElement('option');o.value=cat;o.textContent=cat;el('category').appendChild(o);}
function option(parent,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;parent.appendChild(o);}
for(const name of Object.keys(maps))option(el('mapFilter'),name,name);
function states(){el('stateFilter').replaceChildren();option(el('stateFilter'),'','任意状态');const map=maps[el('mapFilter').value];if(map){const layer=el('layerFilter').value;const j=map.layers.indexOf(Number(layer));const values=Object.values(map.states).flatMap(s=>layer===''?s:[s[j]]);for(const v of [...new Set(values)].sort((a,b)=>a-b))option(el('stateFilter'),v,'S'+v);}filter();}
el('mapFilter').onchange=()=>{el('layerFilter').replaceChildren();option(el('layerFilter'),'','任意层');const map=maps[el('mapFilter').value];if(map)for(const layer of map.layers)option(el('layerFilter'),layer,'L'+layer);states();};el('layerFilter').onchange=states;el('stateFilter').onchange=filter;
function inState(c){const map=maps[el('mapFilter').value],state=el('stateFilter').value,layer=el('layerFilter').value;if(!map||state==='')return true;const seq=map.states[c.sample_id];if(!seq)return false;return layer===''?seq.includes(Number(state)):seq[map.layers.indexOf(Number(layer))]===Number(state);}
function show(c){const d=el('detail');d.replaceChildren();text('h3',c.sample_id+' · '+(c.is_correct?'正确':'错误')+' · '+c.finish_reason,d);
for(const [name,map] of Object.entries(maps)){const s=map.states[c.sample_id];if(!s)continue;text('h4',name+' · '+map.representation+' · '+map.trial_id,d);const row=document.createElement('div');d.appendChild(row);map.layers.forEach((layer,j)=>{const x=text('span','L'+layer+': S'+s[j],row);x.className='tag';});}
for(const [title,key] of [['原始题目','problem'],['标准答案','ground_truth'],['抽取的模型答案','extracted_answer'],['模型完整回答','response_text'],['标准解答','solution'],['实际模型输入（含 chat template）','model_input_text']]){text('h4',title,d);text('pre',c[key],d);} }
function render(){el('list').replaceChildren();const part=filtered.slice(page*25,(page+1)*25);el('count').textContent=filtered.length+' 条 · 第 '+(page+1)+' 页';
for(const c of part){const tr=document.createElement('tr');el('list').appendChild(tr);const td=document.createElement('td');tr.appendChild(td);const b=text('button',c.sample_id,td);b.onclick=()=>show(c);text('td',(c.category||'')+' / '+c.level,tr);text('td',c.is_correct?'✓':'✗',tr);text('td',(c.problem||c.prompt_text||'').slice(0,160),tr);}}
function filter(){const q=el('query').value.toLowerCase(),correct=el('correct').value,cat=el('category').value;filtered=cases.filter(c=>(!q||[c.sample_id,c.problem,c.response_text].join(' ').toLowerCase().includes(q))&&(!correct||String(c.is_correct)===correct)&&(!cat||c.category===cat)&&inState(c));page=0;render();}
el('query').oninput=filter;el('correct').onchange=filter;el('category').onchange=filter;el('prev').onclick=()=>{page=Math.max(0,page-1);render();};el('next').onclick=()=>{if((page+1)*25<filtered.length)page++;render();};render();if(cases.length)show(cases[0]);
</script>
"""


def render_review(study, destination, *, max_trajectories=5000, bootstrap=1000):
    root, dest = Path(study).resolve(), Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    state = json.loads((root / "study.json").read_text())
    cfg = load(root / "configs/mean_gmm.json")
    data = prepare(cfg.data, cfg.execution.cache_root)
    overview = export_cases(data, dest)
    results = declared_results(root, state)
    for name, result in results.items():
        audit = result.validate()
        if not audit["valid"]:
            raise ValueError(f"Invalid trial {name}: {audit['errors']}")
    if any(dest == r.path or r.path in dest.parents for r in results.values()):
        raise ValueError("Review must be outside immutable trials")
    version = stage_version("figure")
    galleries, quality, trajectories = [], [], {}
    primary = [n for n in results if n.startswith(("mean_", "prompt_", "selected_"))]
    for name in sorted(primary):
        r = results[name]
        bundle_root = dest / name
        marker = bundle_root / "manifest.json"
        params = dict(
            max_trajectories=max_trajectories,
            linkage="average",
            silhouette_n=2000,
            seed=42,
            source_version=version,
        )
        cached = False
        if marker.exists():
            old = json.loads(marker.read_text())
            cached = (
                old["parameters"] == params
                and old["inputs"][0]["trial_id"] == r.summary["trial_id"]
                and old["inputs"][0]["summary_sha256"]
                == file_digest(r.path / "summary.json")
                and all(
                    (bundle_root / path).is_file()
                    and file_digest(bundle_root / path) == entry["sha256"]
                    for path, entry in old["outputs"].items()
                )
            )
        if not cached:
            child = FigureBundle(bundle_root, [r], params)
            nodes, edges = state_map(r)
            marginal, profile = dynamics(r)
            scan = selection_surface(r)
            q = cluster_quality(
                r,
                Path(r.config["execution"]["cache_root"])
                / "data"
                / r.summary["snapshot"],
            )
            for label, table in [
                ("nodes", nodes),
                ("edges", edges),
                ("dynamics", profile),
                ("selection", scan),
                ("quality", q),
                ("state_outcomes", state_outcomes(r)),
                ("associations", r.table("associations.csv")),
            ]:
                child.table(label, table)
            title = f"Llama-3.2-1B / MATH / {name}"
            child.figure(
                "entropy_graph", plots.state_graph(nodes, edges, "entropy", title)
            )
            child.figure(
                "correctness_graph",
                plots.state_graph(nodes, edges, "accuracy_delta", title),
            )
            similarity, order = trajectory_similarity(r, max_trajectories)
            child.table("trajectory_order", order)
            child.figure("geometry", plots.geometry(marginal, profile, similarity))
            if (
                r.config["evaluation"]["fixed_k_map"]
                or r.config["cluster"]["k"] is not None
            ):
                child.figure("fixed_K_profile", plots.profile_plot(profile))
            else:
                child.figure("selection", plots.icl_surface(scan))
            child.figure(
                "associations", plots.associations(r.table("associations.csv"))
            )
            child.figure(
                "correctness_bands", plots.correctness_bands(r.table("state_tags.csv"))
            )
            if r.config["cluster"]["method"] == "mfa":
                history = []
                for layer in r.layers:
                    model, _, _ = load_layer(r.path, layer)
                    history.extend(
                        dict(layer=layer, iteration=i + 1, log_likelihood=value)
                        for i, value in enumerate(model.config().get("history", []))
                    )
                if history:
                    trace = pd.DataFrame(history)
                    child.table("em_history", trace)
                    child.figure("em_convergence", plots.em_convergence(trace))
            child.finish(
                status="complete",
                paper_scope="Llama-MATH adaptation; no cross-model claims",
            )
        quality.append(pd.read_csv(bundle_root / "quality.csv").assign(job=name))
        paths = sorted(
            bundle_root / p
            for p in json.loads(marker.read_text())["outputs"]
            if p.endswith(".png")
        )
        title = f"{name} · {r.summary['trial_id']}"
        figures = "".join(
            f'<details><summary>{html.escape(p.stem)} · <a href="{p.stem}.svg">SVG</a></summary><img loading="lazy" src="{p.name}" alt="{p.stem}"></details>'
            for p in paths
        )
        figures = (
            '<p><a href="state_outcomes.csv">逐状态正确率、回答长度与截断比例</a> · 在总览的逐题页选择方法/层/状态可查看成员。</p>'
            + figures
        )
        (bundle_root / "index.html").write_text(
            f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style><main><a href="../index.html">← 总览</a><h1>{html.escape(title)}</h1><p>5,000 real MATH responses · all captured layers · raw features. Same-K control when fixed_k_map is set; MFA has no independent K search in this comparison.</p>{figures}</main>'
        )
        galleries.append(
            f'<li><a href="{name}/index.html">{html.escape(name)}</a> — K={min(x["k"] for x in r.summary["profile"])}–{max(x["k"] for x in r.summary["profile"])}, {len(np.unique(r.states))} observed global states</li>'
        )
        if name.startswith(("mean_", "prompt_")):
            trajectories[name] = dict(
                trial_id=r.summary["trial_id"],
                representation=r.config["data"]["representation"],
                layers=r.layers,
                states={
                    str(sid): s.tolist()
                    for sid, s in zip(r.rows.sample_id, np.asarray(r.states))
                },
            )
    (dest / "trajectories.js").write_text(
        "window.HSS_TRAJECTORIES=" + javascript(trajectories) + ";\n"
    )
    bundle = FigureBundle(
        dest / "comparison",
        results.values(),
        dict(
            bootstrap=bootstrap,
            snapshot=data.info["key"],
            cases_manifest_sha256=file_digest(dest / "cases-manifest.json"),
        ),
    )
    cases = pd.read_parquet(dest / "original_cases.parquet")
    bundle.figure("dataset_overview", plots.dataset_overview(cases))
    for axis in ("category", "level", "finish_reason"):
        if axis in cases:
            table = (
                cases.groupby(axis, dropna=False)
                .is_correct.agg(["count", "sum", "mean"])
                .reset_index()
            )
            bundle.table(
                "correctness_by_" + axis,
                table.rename(
                    columns={"count": "n", "sum": "correct", "mean": "accuracy"}
                ),
            )
    for name, table in collect_tables(list(results.values())).items():
        bundle.table(name, table)
    bundle.table(
        "protocol_inventory",
        pd.DataFrame(
            [
                dict(
                    job=name,
                    observed_global_states=len(np.unique(r.states)),
                    allocated_global_id_span=r.summary["n_global_states"],
                    **r.metadata(),
                )
                for name, r in results.items()
            ]
        ),
    )
    protocols = bundle.path / "configurations.json"
    save_json(protocols, {name: r.config for name, r in results.items()})
    bundle.outputs.append(protocols)
    paired = [
        r
        for name, r in results.items()
        if name in ("mean_gmm", "mean_mfa", "mean_kmeans", "mean_minibatch_kmeans")
    ]
    if len(paired) > 1:
        agreement = compare_trials(paired, all_pairs=True)
        names = {r.summary["trial_id"]: n for n, r in results.items()}
        agreement["reference_job"] = agreement.reference.map(names)
        agreement["target_job"] = agreement.target.map(names)
        bundle.table("method_agreement", agreement)
    quality_frame = pd.concat(quality, ignore_index=True) if quality else pd.DataFrame()
    bundle.table("cluster_quality", quality_frame)
    matched_quality = (
        quality_frame[
            quality_frame.job.isin(
                ["mean_gmm", "mean_mfa", "mean_kmeans", "mean_minibatch_kmeans"]
            )
        ]
        if len(quality_frame)
        else quality_frame
    )
    if len(matched_quality):
        bundle.figure("method_quality", plots.method_quality(matched_quality))
    stability = [r for n, r in results.items() if n.startswith(("stability_", "mean_"))]
    comparisons = []
    for method in ("gmm", "kmeans", "minibatch_kmeans", "mfa"):
        part = [r for r in stability if r.config["cluster"]["method"] == method]
        if len(part) > 1:
            comparisons.append(
                compare_trials(part, all_pairs=True).assign(method=method)
            )
    if comparisons:
        compared = pd.concat(comparisons, ignore_index=True)
        bundle.table("stability", compared)
        bundle.figure(
            "stability",
            plots.control_diagnostics(
                collect_tables(stability)["diagnostics"], compared
            ),
        )
    predictions = [
        r for r in results.values() if r.config["evaluation"]["mode"] == "prediction"
    ]
    if predictions:
        bundle.table(
            "prediction_quality",
            pd.concat([prediction_quality(r) for r in predictions], ignore_index=True),
        )
        bundle.table(
            "prediction_bootstrap",
            pd.concat(
                [bootstrap_predictions(r, bootstrap) for r in predictions],
                ignore_index=True,
            ),
        )
        bundle.figure(
            "prediction",
            plots.prediction_controls(collect_tables(predictions)["evaluation"]),
        )
    bundle.finish(status="complete", study_status=state["status"])
    completed = list(results)
    pending = [
        p.stem for p in (root / "configs").glob("*.json") if p.stem not in results
    ]
    coverage = dict(
        completed=completed,
        pending=pending,
        study_status=state["status"],
        missing_paper_inputs=[
            "Other models and datasets for cross-model Figure 5 and full Table 1",
            "Aligned per-token entropy/logprob for the two monitoring baselines",
        ],
        adaptations="Figures 6 and 9–12 and prediction/monitoring use Llama-MATH here, not paper Qwen runs.",
        updated_at=time.time(),
    )
    pipeline_path = root / "pipeline.json"
    if pipeline_path.exists():
        coverage["pipeline"] = json.loads(pipeline_path.read_text())
    coverage["protocol_differences"] = [
        "MFA uses GMM-selected K; no independent MFA K optimum is claimed.",
        "Monitoring uses K selected on prompt training data, not a full prefix ICL scan.",
        "Fixed-K seed/subsample refits measure centers and assignment stability; they do not test K-selection stability.",
    ]
    save_json(dest / "coverage.json", coverage)
    tables = "".join(
        f'<li><a href="comparison/{p.name}">{p.name}</a></li>'
        for p in sorted((dest / "comparison").glob("*.csv"))
    )
    qtable = ""
    if len(quality_frame):
        summary = quality_frame.groupby("job").agg(
            silhouette=("silhouette", "mean"),
            CH=("calinski_harabasz", "mean"),
            DB=("davies_bouldin", "mean"),
            min_K=("selected_k", "min"),
            max_K=("selected_k", "max"),
            min_active_K=("active_k", "min"),
            not_converged=("converged", lambda v: int(v.eq(False).sum())),
        )
        qtable = (
            '<h2>聚类比较</h2><p>跨层均值仅供概览；请结合逐层 CSV。Silhouette / CH 越高越好，DB 越低越好；不同方法的 ICL 和 silhouette 不能直接比较。active_K 是实际占用的簇数：prompt-last 的常量词嵌入层可以只有一个有效簇。not_converged 表示明确未达到 EM 收敛条件的层数（KMeans 不使用这个标志）；这些结果不能称为已收敛。</p><div class="scroll">'
            + summary.to_html(float_format=lambda x: f"{x:.4f}")
            + "</div>"
        )
    accuracy = overview["correct"] / overview["n"]
    quality_plot = (
        '<div class="card"><img src="comparison/method_quality.png" alt="Matched-K per-layer geometry comparison"></div>'
        if len(matched_quality)
        else ""
    )
    norm_plot = ""
    norm_manifest = dest / "normalization/provenance.json"
    if norm_manifest.exists():
        norm = json.loads(norm_manifest.read_text())
        if norm["post_snapshot"] == data.info["key"] and all(
            file_digest(dest / "normalization" / name) == checksum
            for name, checksum in norm["outputs"].items()
        ):
            norm_plot = '<h2>最后一层 RMSNorm 诊断</h2><p>比较同一道题在前一 block、最后 block 的 pre-RMS 和 post-RMS 生成均值。图中是成对余弦相似度，不是重新聚类；均值归一化也不等于归一化后的 token 均值。</p><div class="card"><img src="normalization/paired_norm_geometry.png" alt="Paired final normalization geometry"></div><p><a href="normalization/paired_norm_geometry.csv">逐题数据</a> · <a href="normalization/summary.csv">汇总统计</a> · <a href="normalization/provenance.json">来源记录</a></p>'
    pipeline_note = html.escape(
        json.dumps(coverage.get("pipeline", {}).get("stages", {}), ensure_ascii=False)
    )
    intro = f"""<!doctype html><meta charset="utf-8"><title>Llama MATH · HSS results</title><style>{CSS}</style><main><header><p class="muted">HIDDEN STATES AS STATES · REAL DATA REVIEW</p><h1>Llama-3.2-1B × MATH 5,000</h1><p>论文对应分析、聚类方法对照与逐题原文。保存时间 {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}。</p><div class="metrics"><div class="metric">{overview["n"]:,} 道题</div><div class="metric">正确率 {accuracy:.2%}</div><div class="metric">{overview["truncated"]} 条达到长度上限</div><div class="metric">{len(completed)} 个实验完成</div></div></header>
<p class="warn">这是单模型 MATH 分析。其他模型和数据集的原论文数值尚不能从这批数据复现。正确性来自 OpenAct 的数学答案评估；关联图是描述统计，预测性能只使用独立测试集。聚类为生成 token 的均值或 prompt 最后 token，不是逐 token 聚类。MFA rank=8 使用 GMM 选出的逐层 K，属于相同 K 对照。</p>
<p><a href="#cases">查看原题与回答</a> · <a href="original_cases.parquet">下载原文与评估 Parquet</a> · <a href="coverage.json">复现范围/进度</a></p>
<div class="card"><img src="comparison/dataset_overview.png" alt="MATH correctness by category, difficulty, and response length"><p class="small">采集结果的描述统计。长度和最终正确性不作为 prompt 预测输入。</p></div>
<p>运行状态：{html.escape(state["status"])}；当前已配置、等待完成：{html.escape(", ".join(pending) or "无")}。阶段：{pipeline_note}</p>
<p class="small">稳定性采用固定 K 的 seed/子集 refit，检验中心与归属稳定性；前缀监测沿用仅在 prompt 训练组选择的 K。这两项为明确配置的扩展对照，不等同于重新扫描 K 的原论文实验。预测表提供多数类准确率、AUROC、PR-AUC、balanced accuracy 和 MCC，避免类别不均衡误导。</p>
{qtable}{quality_plot}{norm_plot}<h2>图集</h2><p>状态图（Fig. 3 / 7 / 8 / 10 / 11）、占用与相似度（Fig. 4）、标签关联（Fig. 6）、选 K 曲线/曲面（Fig. 9）、正确性状态带（Fig. 12）。全部为真实数据；Qwen 图式在这里明确为 Llama 适配。</p><p class="small">最终层使用 post-RMS；与前一层匹配的变化同时包含最后一个 Transformer block 与 RMSNorm 的影响。论文的 ±30 个百分点状态标签是相对全局正确率的固定阈值；当全局正确率低于 30% 时，不可能出现 low 标签，请结合逐状态实际正确率与样本量阅读。</p><ul>{"".join(galleries)}</ul>
<h2>可下载指标</h2><p><a href="comparison/configurations.json">完整实验配置 JSON</a> · method_agreement 中 ARI 表示方法之间分组的一致程度，不是正确性。</p><ul>{tables}</ul><p>每个图集均附 PNG、SVG、CSV 和带哈希的 manifest。采样只用于 silhouette；轨迹相似度使用 {max_trajectories:,} 条上限，导出实际参与的 sample ID。</p>"""
    (dest / "index.html").write_text(intro + VIEWER + "</main>")
    return dict(path=str(dest / "index.html"), overview=overview, **coverage)
