"""Plot audited saved GMM selections against relative layer depth; no fitting."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/three-dataset-gmm-counts-20260926"
LEGACY = ROOT / "results/math-mmlu-gmm-counts-20260925/plot_data.json"
BELEBELE = ROOT / "results/belebele-diagonal-gmm-20260926"
MODELS = {"qwen2": ("Qwen2-7B-Instruct", 28),
          "llama32": ("Llama-3.2-1B-Instruct", 16)}
DATASETS = ("MATH", "MMLU", "BELEBELE")
COLORS = {"MATH": "#0072B2", "MMLU": "#D55E00", "BELEBELE": "#009E73"}
MARKERS = {"MATH": "o", "MMLU": "s", "BELEBELE": "^"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    legacy = read(LEGACY)
    sources = {str(LEGACY.relative_to(ROOT)): sha(LEGACY)}
    for source, expected in legacy["source_hashes"].items():
        assert sha(ROOT / source) == expected, source
        sources[source] = expected
    rows = [dict(r) for r in legacy["rows"]]
    completion = read(BELEBELE / "COMPLETE.json")
    for model in MODELS:
        directory = BELEBELE / model
        audit_path = directory / "independent_audit.json"
        assert sha(audit_path) == completion["independent_audits"][model]
        audit = read(audit_path)
        assert audit["passed"] and audit["n_samples"] == 2700
        selection_path = directory / "selection.json"
        assert sha(selection_path) == audit["hashes"]["selection.json"]
        for path in (audit_path, selection_path):
            sources[str(path.relative_to(ROOT))] = sha(path)
        for r in read(selection_path):
            if r["view"] != "post" or r["tolerance"] not in (0.0, 0.02):
                continue
            assert r["converged"] and not r["upper_boundary_warning"]
            rows.append(dict(model=model, dataset="BELEBELE", n_samples=2700,
                             layer=r["layer"], tolerance=r["tolerance"], k=r["k"],
                             converged=True, fit_path=r["fit_path"],
                             selection_scope="Converged saved adaptive-search candidates; languages pooled"))
    sources[str((BELEBELE / "COMPLETE.json").relative_to(ROOT))] = sha(BELEBELE / "COMPLETE.json")
    for r in rows:
        r["normalized_depth"] = r["layer"] / MODELS[r["model"]][1]
    assert len(rows) == 276
    for model, (_, last) in MODELS.items():
        for dataset in DATASETS:
            for tolerance in (0.0, 0.02):
                group = series(rows, model, dataset, tolerance)
                assert [r["layer"] for r in group] == list(range(last + 1))
                assert group[0]["normalized_depth"] == 0
                assert group[-1]["normalized_depth"] == 1
                assert all(isinstance(r["k"], int) and r["k"] > 0 for r in group)
    assert [(r["model"], r["dataset"], r["layer"], r["tolerance"])
            for r in rows if not r["converged"]] == [("llama32", "MATH", 0, 0.0)]
    return rows, sources


def series(rows, model, dataset, tolerance):
    return sorted((r for r in rows if r["model"] == model and r["dataset"] == dataset
                   and r["tolerance"] == tolerance), key=lambda r: r["layer"])


def plot(rows, tolerance):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), sharey=True)
    fig.subplots_adjust(left=.07, right=.975, bottom=.25, top=.74, wspace=.12)
    percent = int(tolerance * 100)
    fig.text(.07, .949, f"GMM components across depth | {percent}% ICL tolerance",
             size=19, weight="bold", color="#172B3A")
    fig.text(.07, .899, "Full-response raw token means | MATH 5,000; MMLU 14,042; BELEBELE 2,700",
             size=11, color="#475569")
    for ax, (model, (title, last)) in zip(axes, MODELS.items()):
        for dataset in DATASETS:
            group = series(rows, model, dataset, tolerance)
            x = [r["normalized_depth"] for r in group]
            y = [r["k"] for r in group]
            line, = ax.plot(x, y, label=dataset, color=COLORS[dataset], lw=2,
                            marker=MARKERS[dataset], ms=4.2, markeredgewidth=.9,
                            markerfacecolor="white", zorder=3)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            bad = [r for r in group if not r["converged"]]
            if bad:
                ax.scatter([r["normalized_depth"] for r in bad], [r["k"] for r in bad],
                           marker="x", s=100, color="#b91c1c", lw=2.3, zorder=5)
        ax.set_title(f"{title} ({last} blocks)", fontsize=12, pad=13, weight="bold")
        ax.set_xlim(-.018, 1.025)
        ax.set_ylim(0, 120 if tolerance == 0 else 60)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.arange(0, 121, 20) if tolerance == 0 else np.arange(0, 61, 10))
        ax.set_xlabel(r"Normalized layer depth $\ell/L$", labelpad=9)
        ax.axvspan(.991, 1.009, color="#e2e8f0", zorder=0)
        ax.grid(axis="y", color="#e2e8f0", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#94a3b8")
        ax.tick_params(colors="#334155")
    axes[0].set_ylabel("Number of GMM components (K)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.065, .869), ncol=3, frameon=False, fontsize=11,
               columnspacing=3.0, handlelength=2.8)
    fig.text(.07, .132, "Depth 0 = embedding; depth 1 = final post-RMSNorm. Only the depth axis is normalized.",
             size=9.5, color="#475569")
    fig.text(.07, .097, "BELEBELE pools English, German and Chinese (900 each). Existing search and restart budgets differ.",
             size=9.5, color="#475569")
    note = ("Red x: Llama MATH embedding K=36 did not converge in the historical fit."
            if tolerance == 0 else "All selected fits on this 2% plot converged. Vertical scale: 0–60 (0% plot: 0–120).")
    fig.text(.07, .06, note, size=9.5, color="#b91c1c" if tolerance == 0 else "#475569")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"gmm_three_datasets_icl_{percent:02d}.{ext}", dpi=190, facecolor="white")
    return fig


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "font.size": 11})
    rows, sources = collect()
    payload = dict(metric="Selected diagonal GMM component count K, not occupancy",
                   x_axis="stored hidden-state index / number of transformer blocks",
                   representation="Raw full-response mean; final point includes model RMSNorm",
                   tolerances=[0, .02], no_refitting=True,
                   selection_rule="Smallest K satisfying ICL <= best + tolerance * max(abs(best),1)",
                   limitations=["Dataset size, candidate searches and restart budgets differ.",
                                "Llama MATH embedding strict-ICL fit did not converge; preserved and marked.",
                                "Final pre-RMSNorm fits are excluded; final plotted point is post-RMSNorm."],
                   source_hashes=sources, rows=rows)
    (OUT / "plot_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with PdfPages(OUT / "gmm_three_datasets_both_tolerances.pdf") as pdf:
        for tolerance in (0.0, .02):
            fig = plot(rows, tolerance)
            pdf.savefig(fig)
            plt.close(fig)
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>三个数据集：归一化层深与GMM簇数</title>",
             "<style>body{max-width:1300px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#172b3a}img{width:100%;height:auto}a{color:#0072b2}table{border-collapse:collapse}td,th{padding:4px 16px;border-bottom:1px solid #ddd}.note{background:#f1f5f9;padding:14px;border-radius:8px}</style>",
             "<h1>三个数据集：归一化层深与 GMM 簇数</h1>",
             "<p>左侧Qwen2-7B-Instruct，右侧Llama-3.2-1B-Instruct。横轴为层号÷模型总层数，纵轴为GMM成分数K。蓝色MATH、橙色MMLU、绿色BELEBELE。</p>",
             "<p class='note'>只把层深换算到0–1，hidden states和K均未归一化。每个数据集均使用全量回答的raw token mean。0%仍然使用ICL选择，只是不设容差；2%在允许的ICL范围内选最小K。两张图纵轴分别为0–120与0–60。</p>",
             "<p>MATH 5,000条；MMLU 14,042条；BELEBELE为英德中合并2,700条。拟合搜索及初始化预算不同，这是已保存地图的描述性对比。最后一点包含模型自身最终RMSNorm，未混入单独的pre-RMS拟合。</p>",
             "<p style='color:#b91c1c'>Llama MATH的0%版本在embedding层选中K=36的旧拟合没有收敛，图中用红叉标注；其余展示点均收敛。</p>",
             "<p><a href='gmm_three_datasets_both_tolerances.pdf'>下载两页PDF</a> · <a href='plot_data.json'>全部276个数值及来源哈希</a></p>"]
    for percent in (0, 2):
        stem = f"gmm_three_datasets_icl_{percent:02d}"
        parts.extend([f"<h2>{percent}% ICL容差</h2><img src='{stem}.png' alt='{percent}% ICL tolerance, Qwen and Llama, three datasets'>",
                      f"<p><a href='{stem}.pdf'>PDF</a> · <a href='{stem}.svg'>SVG</a> · <a href='{stem}.png'>PNG</a></p>"])
    for model, (title, last) in MODELS.items():
        parts.extend([f"<details><summary>{title}逐层数值</summary><table><tr><th>层</th><th>层深</th>",
                      "".join(f"<th>{d} {int(t*100)}%</th>" for d in DATASETS for t in (0.0, .02)) + "</tr>"])
        for layer in range(last + 1):
            selected = [next(r for r in rows if r['model'] == model and r['dataset'] == d
                             and r['tolerance'] == t and r['layer'] == layer)
                        for d in DATASETS for t in (0.0, .02)]
            parts.append(f"<tr><td>{layer}</td><td>{layer/last:.4f}</td>" +
                         "".join("<td>"+html.escape(str(r['k']) + (" ×" if not r['converged'] else ""))+"</td>" for r in selected) + "</tr>")
        parts.append("</table></details>")
    parts.append("</html>")
    (OUT / "index.html").write_text("\n".join(parts))
    print(json.dumps(dict(output=str(OUT), validated_points=len(rows), source_files=len(sources))))


if __name__ == "__main__":
    main()
