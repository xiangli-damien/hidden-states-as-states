"""Paper-defined, label-free trajectory scores under a paired final-norm ablation.

The final pre-RMS mean is mean_t(pre(h_t)), never RMSNorm(mean_t(h_t)).
All primary trajectories include slot zero (embeddings). Score orientation is
fixed by the manuscript: higher NDR and CoE predict correct; no test-set flips.
"""
import html
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from hss.analysis.channel_data import ChannelData, load_config
from hss.analysis.confidence_study import partitions
from hss.analysis.confidence_bootstrap import auc_samples
from hss.experiments.artifacts import save_json, file_digest, runtime_versions

NAMES = {"llama32_math": "Llama-3.2-1B / MATH", "qwen2_math": "Qwen2-7B / MATH",
         "llama32_mmlu": "Llama-3.2-1B / MMLU"}
METRICS = ["ndr", "coe_r", "coe_c", "negative_final_norm"]
LABELS = ["NDR (+)", "CoE-R (+)", "CoE-C (+)", "Final norm (-)"]


def trajectory_scores(states):
    """Equations 3, 4, 7; float64 arithmetic, batch-sized temporary arrays."""
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [question, embedding+layers, hidden] means")
    norms = np.linalg.norm(x, axis=2)
    displacement = np.linalg.norm(x[:, -1] - x[:, 0], axis=1)
    cos = np.einsum("nld,nld->nl", x[:, 1:], x[:, :-1]) / np.maximum(norms[:, 1:] * norms[:, :-1], 1e-30)
    angles = np.arccos(np.clip(cos, -1, 1))
    endpoint_cos = np.einsum("nd,nd->n", x[:, 0], x[:, -1]) / np.maximum(norms[:, 0] * norms[:, -1], 1e-30)
    endpoint_angle = np.arccos(np.clip(endpoint_cos, -1, 1))
    # Degenerate denominators are rejected, never silently turned into scores.
    if np.any(norms <= 1e-12) or np.any(displacement <= 1e-6) or np.any(endpoint_angle <= 1e-6):
        raise ValueError("Degenerate trajectory norm/endpoint displacement/angle")
    magnitudes = np.linalg.norm(np.diff(x, axis=1), axis=2) / displacement[:, None]
    coe_r = (magnitudes - angles / endpoint_angle[:, None]).mean(1)
    coe_c = np.hypot((magnitudes * np.cos(angles)).mean(1), (magnitudes * np.sin(angles)).mean(1))
    update = x[:, -1] - x[:, -2]
    update_norm = np.linalg.norm(update, axis=1)
    return {"ndr": norms.mean(1) / norms[:, -1], "coe_r": coe_r, "coe_c": coe_c,
            "negative_final_norm": -norms[:, -1], "final_norm": norms[:, -1],
            "last_to_previous_norm_ratio": norms[:, -1] / norms[:, -2],
            "last_update_cosine": np.einsum("nd,nd->n", update, x[:, -2]) / np.maximum(update_norm * norms[:, -2], 1e-30),
            "min_endpoint_angle": endpoint_angle, "endpoint_displacement": displacement}


def within_length_auc(y, score, bins):
    """Pair-weighted AUC using only pairs in the same discovery-defined length bin.

    This is a coarse sensitivity check, not exact matching/causal adjustment.
    """
    total = 0
    wins = 0.
    for b in np.unique(bins):
        mask = bins == b
        positives = int(y[mask].sum())
        pairs = positives * (int(mask.sum()) - positives)
        if pairs:
            wins += pairs * roc_auc_score(y[mask], score[mask])
            total += pairs
    return float(wins / total) if total else None


def group_summary(values, y, cfg):
    a, b = values[y == 1], values[y == 0]
    rng = np.random.default_rng(cfg["seed"])
    differences = np.asarray([rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
                              for _ in range(cfg["bootstrap"])])
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return {"correct_mean": float(a.mean()), "incorrect_mean": float(b.mean()),
            "correct_median": float(np.median(a)), "incorrect_median": float(np.median(b)),
            "difference": float(a.mean()-b.mean()), "difference_ci": np.quantile(differences,[.025,.975]).tolist(),
            "correct_over_incorrect": float(a.mean()/b.mean()) if b.mean() != 0 else None,
            "cohen_d": float((a.mean()-b.mean())/pooled) if pooled > 0 else None}


def evaluate(rows, train, test, scores, cfg):
    y = rows.y.to_numpy(int)
    length = rows.n_tokens.to_numpy(float)
    edges = np.unique(np.quantile(length[train], np.linspace(0, 1, cfg["length_bins"]+1)))[1:-1]
    bins = np.searchsorted(edges, length, side="right")
    clean = ~(rows.truncated | rows.parse_failed).to_numpy(bool)
    result = {"n": len(rows), "confirmation_n": len(test), "correct_n": int(y.sum()),
              "truncated_n": int(rows.truncated.sum()), "parse_failed_n": int(rows.parse_failed.sum()),
              "length_bin_edges_from_discovery": edges.tolist(), "scores": {}, "paired_pre_minus_post": {}}
    draws = {}
    for view in ["post", "pre"]:
        result[view] = {key: group_summary(scores[f"{view}_{key}"].to_numpy(), y, cfg)
                        for key in ["final_norm", "last_to_previous_norm_ratio", "last_update_cosine"]}
        result[view]["fraction_last_norm_contracts"] = float((scores[f"{view}_last_to_previous_norm_ratio"] < 1).mean())
        for metric in METRICS:
            key = f"{view}_{metric}"
            s = scores[key].to_numpy()
            draws[key] = auc_samples(y[test], s[test], cfg["seed"], cfg["bootstrap"])
            fpr, tpr, _ = roc_curve(y[test], s[test])
            keep = test[clean[test]]
            result["scores"][key] = {
                "auc": float(roc_auc_score(y[test], s[test])),
                "ci": np.quantile(draws[key], [.025,.975]).tolist(),
                "average_precision": float(average_precision_score(y[test], s[test])),
                "fpr95": float(fpr[np.flatnonzero(tpr >= .95)[0]]),
                "all_auc": float(roc_auc_score(y, s)),
                "clean_auc": float(roc_auc_score(y[keep], s[keep])), "clean_n": len(keep),
                "within_length_bin_auc": within_length_auc(y[test], s[test], bins[test]),
                "spearman_response_length": float(spearmanr(s[test], length[test]).statistic)}
    for metric in METRICS:
        delta = draws[f"pre_{metric}"] - draws[f"post_{metric}"]
        result["paired_pre_minus_post"][metric] = {
            "difference": result["scores"][f"pre_{metric}"]["auc"] - result["scores"][f"post_{metric}"]["auc"],
            "ci": np.quantile(delta, [.025,.975]).tolist()}
    result["negative_length_auc"] = float(roc_auc_score(y[test], -length[test]))
    result["previous_layer_norm"] = group_summary(scores.previous_layer_norm.to_numpy(), y, cfg)
    return result


def figures(out, name, profiles, rows, scores, result):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    y = rows.y.to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    colors = {"pre": "#277c77", "post": "#b34e28"}
    for view in ["pre", "post"]:
        norms = profiles[view]
        diff = norms[y == 1].mean(0) - norms[y == 0].mean(0)
        se = np.sqrt(norms[y == 1].var(0, ddof=1)/(y == 1).sum() + norms[y == 0].var(0, ddof=1)/(y == 0).sum())
        layer = np.arange(norms.shape[1])
        axes[0].plot(layer, diff, ".-", color=colors[view], label=f"Final {view}-RMS")
        axes[0].fill_between(layer, diff-1.96*se, diff+1.96*se, color=colors[view], alpha=.12)
        g = result[view]["final_norm"]
        for label in [0, 1]:
            a = np.sort(scores.loc[y == label, f"{view}_final_norm"].to_numpy())
            # Separate final-norm scales deserve separate x axes, not pooled violins.
            ax = axes[1 if view == "pre" else 2]
            ax.plot(a, np.arange(1, len(a)+1)/len(a), label="Correct" if label else "Incorrect",
                    color="#287348" if label else "#bd553d")
        ax.set(title=f"Final {view}-RMS: correct / incorrect = {g['correct_over_incorrect']:.3f}",
               xlabel="Norm of response mean", ylabel="Cumulative fraction")
        ax.legend()
    axes[0].axhline(0, color="gray", lw=.8)
    axes[0].set(title="Correct minus incorrect mean norm", xlabel="Layer (0 = embedding)", ylabel="Norm difference")
    axes[0].legend()
    fig.suptitle(NAMES[name] + f" | all {len(rows):,} unique questions; response hidden means")
    for ext in ["png", "svg"]: fig.savefig(out/f"norms.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained", sharey=True)
    xx = np.arange(len(METRICS))
    for view, offset in [("post", -.10), ("pre", .10)]:
        values = [result["scores"][f"{view}_{k}"] for k in METRICS]
        mean = np.array([v["auc"] for v in values])
        ci = np.array([v["ci"] for v in values])
        axes[0].errorbar(xx+offset, mean, yerr=[mean-ci[:,0], ci[:,1]-mean], fmt="o", capsize=4,
                         color=colors[view], label=f"Final {view}-RMS")
        axes[1].plot(xx+offset, [v["within_length_bin_auc"] for v in values], "o", color=colors[view], label=f"Final {view}-RMS")
    for ax in axes:
        ax.set_xticks(xx, LABELS)
        ax.axhline(.5, color="gray", ls="--", lw=1)
        ax.set_ylim(.3, .9)
        ax.set_ylabel("AUROC (paper's fixed direction)")
        ax.legend()
    axes[0].set_title(f"Held-out questions (n={result['confirmation_n']:,}); 95% bootstrap CI")
    axes[1].set_title("Compare only pairs in the same length bin\nPoint estimates; coarse length sensitivity")
    fig.suptitle(NAMES[name])
    for ext in ["png", "svg"]: fig.savefig(out/f"auc.{ext}", dpi=160)
    plt.close(fig)


def render(root, results):
    tables, norms = [], []
    for name, r in results.items():
        for metric in METRICS:
            row = {"Dataset": NAMES[name], "Metric": metric}
            for view in ["post", "pre"]:
                s = r["scores"][f"{view}_{metric}"]
                row[view+" AUROC [95% CI]"] = f"{s['auc']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
            row["pre − post"] = f"{r['paired_pre_minus_post'][metric]['difference']:+.3f}"
            tables.append(row)
        for view in ["pre", "post"]:
            s = r[view]["final_norm"]
            norms.append({"Dataset":NAMES[name],"Final layer":view,"Correct mean":s["correct_mean"],
                          "Incorrect mean":s["incorrect_mean"],"Correct / incorrect":s["correct_over_incorrect"],
                          "Difference":s["difference"],"Cohen d":s["cohen_d"]})
    pd.DataFrame(tables).to_csv(root/"headline.csv", index=False)
    pd.DataFrame(norms).to_csv(root/"final_norms.csv", index=False)
    intro = """<h1>NDR / CoE：把末层换成 pre-RMSNorm 后</h1>
<p>只使用已采集的完整 response hidden mean。embedding 和中间层保持不变，仅把最后一层 post-RMS 均值替换为 pre-RMS 均值。无需重新生成。</p>
<p><b>公式：</b>NDR = 所有层（含 embedding）的向量范数平均 / 最后一层范数；CoE-R、CoE-C 严格按稿件公式 3、4。三个分数均固定“越高越正确”，不根据测试结果翻转符号。</p>
<p>范数指 ||mean_t(h_t)||，不是 mean_t(||h_t||)。pre 与 post 分别先逐 token 读取，再各自求均值；没有把均值输入 RMSNorm。包括所有已保存的生成 token（含 EOS 等特殊 token），不含 prompt。</p>
<p>评分没有训练。AUROC 使用原有 60% 验证分区；此分区先前已分析过，所以本次是探索性复核，不是新的盲测。95% CI 为 1,000 次按正确/错误分层、成对重采样。范数组均值使用全部去重问题；曲线阴影是均值差的正态近似 95% 区间。</p>
<p>MATH 每模型 5,000 条；MMLU 原始 14,042 条，移除 105 条重复 prompt 后 13,937 条。当前 Llama 是 3.2-1B，不是稿件主实验的 Llama-3-8B；提示与生成协议也是当前 OpenAct 版本。因此这是 pre/post 配对检验，不是原论文数值的完全复现。</p>
<p>长度敏感性：只比较同一长度 bin 内的正确/错误对，bin 边界由 40% 发现集确定，按可比较对数加权。它只做粗粒度控制，不能声称完全排除长度影响。analysis.json 另含排除截断/解析失败后的结果、AUPR、FPR95、长度相关性和末层实际更新的 cosine。</p>
<p><b>机制边界：</b>原生 hidden_states 最后一个槽经过 RMSNorm；post[L] − hidden[L−1] 同时包含末层 block 更新与归一化，不能直接当作 block 的残差更新。pre[L] − hidden[L−1] 才能用于检验 block 更新的收缩/反对齐。</p>"""
    body = intro + "<h2>AUROC</h2>" + pd.DataFrame(tables).to_html(index=False, escape=True)
    body += "<h2>最后一层范数</h2>" + pd.DataFrame(norms).to_html(index=False,float_format=lambda x:f"{x:.4f}")
    for name, r in results.items():
        body += f'<h2>{html.escape(NAMES[name])}</h2><img src="{name}/norms.png"><img src="{name}/auc.png">'
        body += f'<p><a href="{name}/samples.parquet">逐题分数</a> · <a href="{name}/analysis.json">完整统计</a> · <a href="{name}/norm_profiles.csv">各层范数</a></p>'
    body += '<p><a href="protocol.md">协议与复现</a> · <a href="provenance.json">数据与代码版本</a> · <a href="headline.csv">AUROC 表</a> · <a href="final_norms.csv">范数表</a></p>'
    (root/"index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>NDR / CoE pre-norm</title><style>body{max-width:1320px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#182b38}h1,h2{color:#174b54}img{width:100%;margin:12px 0}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd5da;text-align:left}tr:nth-child(even){background:#f4f7f8}a{color:#176872}</style>'+body+'</html>')


def run(cfg):
    start = time.monotonic()
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    cc = load_config(cfg["channel_config"])
    results, identities = {}, {}
    for name in cfg["datasets"]:
        data = ChannelData(cc, name)
        train, test, saved = partitions(cfg, name, data.rows)
        means = data.array("mean")
        pre_mean = data.array("pre_mean")
        if means.shape[0] != len(saved) or pre_mean.shape != means[:, -1].shape:
            raise ValueError("Final pre/post dimensions or identity mismatch")
        out = root/name
        out.mkdir(exist_ok=True)
        scores = data.rows.copy()
        scores["partition"] = saved.partition.to_numpy()
        profiles = {}
        for view in ["post", "pre"]:
            batches, norm_batches, no_embed = [], [], []
            for a in range(0, len(means), cfg["batch_size"]):
                x = means[a:a+cfg["batch_size"]].astype(np.float64)
                if view == "pre": x[:, -1] = pre_mean[a:a+len(x)]
                batches.append(pd.DataFrame(trajectory_scores(x)))
                no_embed.append(pd.DataFrame(trajectory_scores(x[:, 1:])))
                norm_batches.append(np.linalg.norm(x, axis=2))
            profiles[view] = np.concatenate(norm_batches)
            batch = pd.concat(batches, ignore_index=True)
            for col in batch: scores[f"{view}_{col}"] = batch[col].to_numpy()
            reduced = pd.concat(no_embed, ignore_index=True)
            for metric in METRICS: scores[f"{view}_no_embedding_{metric}"] = reduced[metric].to_numpy()
        scores["previous_layer_norm"] = profiles["pre"][:, -2]
        assert np.array_equal(profiles["post"][:, :-1], profiles["pre"][:, :-1])
        result = evaluate(data.rows, train, test, scores, cfg)
        result["embedding_excluded_sensitivity_auc"] = {f"{v}_{m}":float(roc_auc_score(scores.y.iloc[test], scores[f"{v}_no_embedding_{m}"].iloc[test])) for v in ["pre","post"] for m in METRICS}
        results[name] = result
        identities[name] = {"cache": data.info, "raw_n":data.raw_n,"analytic_n":len(data.rows),
                            "frozen_partition_sha256":file_digest(Path(cfg["prior_root"])/name/"samples.parquet")}
        scores.to_parquet(out/"samples.parquet", index=False)
        save_json(out/"analysis.json", result)
        profile_rows = []
        for view, values in profiles.items():
            for layer in range(values.shape[1]):
                for label in [0,1]:
                    vals = values[data.rows.y.eq(label),layer]
                    profile_rows.append({"view":view,"layer":layer,"correct":label,"n":len(vals),"mean":float(vals.mean()),"std":float(vals.std(ddof=1))})
        pd.DataFrame(profile_rows).to_csv(out/"norm_profiles.csv",index=False)
        figures(out,name,profiles,data.rows,scores,result)
        print(json.dumps({"dataset":name,"n":len(data.rows),"pre_final_norm":result["pre"]["final_norm"],
                          "post_final_norm":result["post"]["final_norm"],"scores":result["scores"]}),flush=True)
        del means, pre_mean, profiles
    save_json(root/"analysis.json",results)
    save_json(root/"provenance.json",{"config":cfg,"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "source_sha256":file_digest(Path(__file__)),"runtime":runtime_versions(),"datasets":identities,
              "seconds":time.monotonic()-start})
    protocol = Path("docs/mean-geometry.zh-CN.md")
    (root/"protocol.md").write_text(protocol.read_text())
    render(root, results)

