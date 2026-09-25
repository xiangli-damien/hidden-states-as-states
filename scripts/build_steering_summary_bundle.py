"""Export audited steering summaries and figures; never run models or rescore answers."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs/steering-report-20260925"
REPORT = ROOT / "docs/steering-complete-report-20260925.zh-CN.md"
OUT = RESULTS / "steering-complete-20260925"
EXPECTED = {
    "sink-direction-20260925/summary.json": "dfc517abafb2dfcf1dab0f63669196b67339025d411e9267bf42696df67e8d0b",
    "sink-followup-20260925/plan.json": "62739d48d57d9a35983e49014a596c06e5f047b7904328115f5d63ef9242e435",
    "sink-followup-20260925/step2_screen_long.parquet": "e8c92f8b65abf09fc228851c5bfc0d230d3433518d4089bffe76b1c6a3c83db3",
    "revision-completion-20260924/steering_summary.json": "c22e0fae0b5810b9281c25ab915996dba89d6a452aaea92cea853ee5039e81aa",
    "projection-first-v2-20260924/summary.json": "b3f46f4bd02031f40ce0a3963b66dfd0daaf8fc231030249c869a3c7458637e2",
    "projection-first-v2-20260924/scoring_reviewed_sensitivity_v2/summary.json": "43464782a738d8e7fb21cc271aea73c8bd4b0d4865e9e0e25afedfe9dc0e1837",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(rel):
        path = RESULTS / rel
        hashes[rel] = digest(path)
        return json.loads(path.read_text())

    for rel, expected in EXPECTED.items():
        hashes[rel] = digest(RESULTS / rel)
        assert hashes[rel] == expected, (rel, "source changed")

    pilot = read("sink-direction-20260925/summary.json")
    decision = read("sink-followup-20260925/step2_decision.json")
    step1 = read("sink-followup-20260925/step1_summary.json")
    step3 = read("sink-followup-20260925/step3_status.json")
    original = read("revision-completion-20260924/steering_summary.json")
    extension = read("projection-first-v2-20260924/summary.json")
    review = read("projection-first-v2-20260924/scoring_reviewed_sensitivity_v2/summary.json")
    # Preserve all aggregate comparisons but omit bulky question-level records.
    comparisons = [{k: v for k, v in c.items() if k != "pairs"} for c in review["comparisons"]]
    metrics = {
        "scope": "Qwen2-7B-Instruct steering; frozen results through 2026-09-25; no new fitting or rescoring",
        "pilot": {k: pilot[k] for k in ("complete", "selection", "results", "paired", "predeclared_success", "limitations")},
        "followup_step1": step1,
        "followup_decision": decision,
        "followup_step3": step3,
        "original_projection": {k: original[k] for k in ("rows", "paired_controls", "unit", "limitation")},
        "extension_comparisons": [{k: v for k, v in c.items() if k != "per_question"} for c in extension["comparisons"]],
        "partial_AI_review": {**{k: review[k] for k in ("full_semantic_rescoring_complete", "independent_human_review", "records", "AI_reviewed_records", "AI_reviewed_unique_answers", "unreviewed_records", "limitations")}, "comparisons": comparisons},
        "source_sha256": hashes,
    }
    # Check every displayed ablation count against the saved aggregates.
    expected_original = [(13, 15), (13, 14), (13, 16), (13, 17), (13, 14), (13, 17), (16, 16), (15, 18), (13, 16), (13, 18)]
    expected_review = [(15, 19), (15, 18), (15, 18), (15, 18), (15, 15), (15, 20), (17, 18), (16, 19), (15, 19), (15, 20)]
    for i in range(1, 11):
        method = f"D{i:02d}"
        control = "baseline_same_trigger" if i in (7, 8) else "baseline"
        a = next(c for c in comparisons if c["method"] == method and c["control"] == control)
        b = a["partial_AI_review"]
        assert (round(b["control_accuracy"] * b["n"]), round(b["method_accuracy"] * b["n"])) == expected_review[i - 1]
        b = a["original"]
        assert (round(b["control_accuracy"] * b["n"]), round(b["method_accuracy"] * b["n"])) == expected_original[i - 1]
    assert not decision["pass_set"]
    (ASSETS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
    colors = ["#61778A", "#166A80", "#C18032"]
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, cohort, title in zip(axs, ("sink_test", "normal_test"), ("Historical sink questions (n=55)", "Normal-question control (n=100)")):
        for i, (arm, label) in enumerate((("zero", "Zero"), ("hss", "HSS direction"), ("random", "Random direction"))):
            row = pilot["results"][cohort][arm]
            vals = [row["boxed_rate"] * 100, row["correct_rate"] * 100]
            bars = ax.bar(np.arange(2) + (i - 1) * .24, vals, .22, color=colors[i], label=label)
            ax.bar_label(bars, labels=[f'{row["boxed_count"]}/{row["n"]}', f'{row["correct_count"]}/{row["n"]}'], padding=3, fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["Complete boxed answer", "Automatic correctness"], ylim=(0, 100), ylabel="Share of questions (%)", title=title)
        ax.grid(axis="y", alpha=.14)
        ax.set_axisbelow(True)
    axs[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Sustained block14 direction: alpha=0.3 selected on separate dev16", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "sink-results.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    names = ["C0  Fixed direction", "C1  Clamp excess sink-axis component", "C2  Type-specific correct-answer direction", "C3  Direction projected into local chart"]
    for i, key in enumerate(("C0", "C1", "C2", "C3")):
        row = decision["per_candidate"][key]
        m, (lo, hi) = row["mean_d_ans"], row["ci95_d_ans"]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color="#166A80", capsize=5, markersize=7, lw=2)
        ax.text(.22, i, f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]", va="center", fontsize=9)
    ax.axvline(0, color="#7E8490", ls="--", lw=1)
    ax.set(yticks=range(4), yticklabels=names, xlim=(-.21, .47), xlabel="Change in summed gold-answer log probability (nats)", title="Follow-up screen: 97 sink questions; paired 95% bootstrap intervals")
    ax.invert_yaxis()
    ax.text(0, -.25, "All intervals include zero. No candidate passed the frozen screening rules.", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "followup-screen.png", bbox_inches="tight")
    plt.close(fig)

    original_row = next(r for r in original["rows"] if r["stage"] == "test" and r["method"] == "c1_1.0")
    selected = [next(c for c in comparisons if c["group"] == group and c["method"] == "local8" and c["control"] == "baseline")["partial_AI_review"] for group in ("MATH_primary256_supplement", "GSM8K_frozen_transfer")]
    estimates = [original_row["delta_accuracy"]["estimate"]] + [c["delta"] for c in selected]
    cis = [original_row["delta_accuracy"]["ci95"]] + [c["ci95"] for c in selected]
    names = ["MATH256 / frozen original scoring", "MATH256 / partial AI-review sensitivity", "GSM64 / partial AI-review sensitivity"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (m, (lo, hi)) in enumerate(zip(estimates, cis)):
        m, lo, hi = np.array([m, lo, hi]) * 100
        ax.errorbar(m, i, xerr=[[m-lo], [hi-m]], fmt="o", capsize=5, color="#166A80" if i == 0 else "#C18032", lw=2)
        ax.text(5.6, i, f"{m:+.2f} pp", va="center")
    ax.axvline(0, color="#7E8490", ls="--", lw=1)
    ax.set(yticks=range(3), yticklabels=names, xlim=(-6.8, 8), xlabel="Local8 minus baseline accuracy (percentage points)", title="Local projection: main test and scoring sensitivity")
    ax.invert_yaxis()
    ax.text(0, -.27, "Same questions across MATH scoring versions; do not pool. Partial AI review is not a human gold standard.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "projection-test.png", bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(REPORT, OUT / "steering-complete-report.zh-CN.md")
    shutil.copytree(ASSETS, OUT / ASSETS.name, dirs_exist_ok=True)
    # Include the later energy-matched follow-up and all its relative links.
    for name in ["sink-energy-matched-results-20260925.zh-CN.md", "sink-energy-matched-protocol-20260925.zh-CN.md"]:
        if (ROOT / "docs" / name).exists():
            shutil.copy2(ROOT / "docs" / name, OUT / name)
    extra = ROOT / "docs/sink-energy-matched-20260925"
    if extra.exists():
        extra_audit = json.loads((extra / "audit_SUCCESS.json").read_text())
        assert extra_audit["passed"] and extra_audit["summary_sha256"] == digest(extra / "summary.json")
        shutil.copytree(extra, OUT / extra.name, dirs_exist_ok=True)
    manifest = {p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    bundle = OUT.parent / "steering-complete-20260925.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, arcname=p.relative_to(OUT))
    print(json.dumps({"sources_verified": len(EXPECTED), "ablations_verified": 10, "bundle": str(bundle), "bytes": bundle.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
