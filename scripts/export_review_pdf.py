"""Create a read-only figure book from a verified review (requires reportlab)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.review), Path(args.output)
    coverage = json.loads((root / "coverage.json").read_text())
    cases = json.loads((root / "cases-manifest.json").read_text())
    images = []
    for group in ["comparison"] + sorted(coverage["completed"]):
        marker = root / group / "manifest.json"
        if not marker.exists():
            continue
        manifest = json.loads(marker.read_text())
        for name, entry in sorted(manifest["outputs"].items()):
            if name.endswith(".png"):
                path = root / group / name
                if sha(path) != entry["sha256"]:
                    raise ValueError(f"Figure checksum mismatch: {path}")
                images.append((group, path, sha(path)))
    marker = root / "normalization/provenance.json"
    if marker.exists():
        norm = json.loads(marker.read_text())
        if norm["post_snapshot"] != cases["snapshot"]:
            raise ValueError("Normalization figure has a different data snapshot")
        path = marker.parent / "paired_norm_geometry.png"
        if sha(path) != norm["outputs"][path.name]:
            raise ValueError("Normalization figure checksum mismatch")
        images.append(("normalization", path, sha(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = landscape(A4)
    pdf = Canvas(str(output), pagesize=(width, height))
    pdf.setTitle("Llama-3.2-1B MATH - HSS figure review")
    pdf.setAuthor("Xiang Li")
    total = len(images) + 2
    blue, grey = HexColor("#183b56"), HexColor("#536473")

    def footer(page):
        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(grey)
        pdf.drawString(
            32,
            19,
            "Snapshot figure book | SVG/CSV and original responses: companion index.html",
        )
        pdf.drawRightString(width - 32, 19, f"{page} / {total}")

    def paragraph(text, y, size=10):
        p = Paragraph(
            text,
            ParagraphStyle(
                "note",
                fontName="Helvetica",
                fontSize=size,
                leading=size * 1.4,
                textColor=grey,
            ),
        )
        _, h = p.wrap(width - 64, height)
        p.drawOn(pdf, 32, y - h)
        return y - h - 12

    pdf.setFillColor(blue)
    pdf.setFont("Helvetica-Bold", 25)
    pdf.drawString(32, height - 54, "Llama-3.2-1B / MATH")
    pdf.setFont("Helvetica", 14)
    pdf.drawString(32, height - 80, "Hidden States as States - real-data figure review")
    y = paragraph(
        f"{cases['n']:,} responses | {cases['correct']:,} correct ({cases['correct'] / cases['n']:.2%}) | {cases['truncated']:,} length-truncated | {len(coverage['completed'])} completed experiments",
        height - 103,
        12,
    )
    y = paragraph(
        "Scope: raw 2,048-dimensional features; 17 captured layer positions including embedding; final post-RMS. Response-mean trajectories summarize complete generations, not token-time trajectories. MFA rank 8 uses GMM-selected K; this is a matched-K comparison.",
        y,
    )
    qpath = root / "comparison/cluster_quality.csv"
    if qpath.exists():
        rows = list(csv.DictReader(qpath.open()))
        table = [
            ["Experiment", "K range", "Silhouette", "CH", "DB", "EM not converged"]
        ]
        for job in sorted({r["job"] for r in rows}):
            part = [r for r in rows if r["job"] == job]
            values = []
            for field in ("silhouette", "calinski_harabasz", "davies_bouldin"):
                nums = [
                    float(r[field])
                    for r in part
                    if r[field] and math.isfinite(float(r[field]))
                ]
                values.append(f"{sum(nums) / len(nums):.4f}" if nums else "-")
            ks = [int(r["selected_k"]) for r in part]
            table.append(
                [
                    job,
                    f"{min(ks)}-{max(ks)}",
                    *values,
                    str(sum(r["converged"].lower() == "false" for r in part)),
                ]
            )
        item = Table(table, colWidths=[195, 66, 90, 85, 85, 150], repeatRows=1)
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), blue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#edf3f7"), white]),
                ]
            )
        )
        _, h = item.wrap(width - 64, height)
        item.drawOn(pdf, 32, y - h)
        y -= h + 15
    y = paragraph(
        "Metrics above are descriptive means across layers. Silhouette/CH favor compact Euclidean groups; they do not establish predictive value. EM iteration-limit results are explicitly marked and must not be described as converged. KMeans has no EM convergence flag.",
        y,
    )
    footer(1)
    pdf.showPage()
    pdf.setFillColor(blue)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(32, height - 54, "Reading guide and reproduction scope")
    y = paragraph(
        "This snapshot does not reproduce all manuscript numbers. Other-model/dataset experiments and unavailable token entropy/logprob baselines are excluded. The HTML coverage table lists completed adaptations and pending stages. Each image comes from checksum-verified saved results; schematic pages contain no measurements.",
        height - 90,
    )
    for note in [
        "Method comparison: GMM uses diagonal covariance and ICL over K=2..80, with the smallest K in a 2% near-optimal band. The mean_* controls use those same per-layer K values. selected_kmeans and selected_minibatch_kmeans instead select their own K by sampled silhouette. Correctness labels never select clusters or K.",
        "Interpretation: correctness graphs describe completed responses. A state's outcome can reflect problem type, response length and truncation. Inspect state_outcomes.csv and the original responses before assigning a semantic interpretation. The paper's relative +/-30 percentage-point low-correctness tag cannot occur when overall accuracy is below 30%.",
        "Evaluation stages: prompt prediction refits on 40% training responses and evaluates on 60% held out. Prefix monitoring uses response-level 40/20/40 splits and a validation-calibrated 10% FAR target. Its K comes from prompt training data. Fixed-K seed/subsample refits test center and assignment stability, not K-selection stability.",
        "Original cases: open the companion index.html, then filter by method, layer and state. Each case contains the original question, full response, gold solution, extracted answer, correctness and rendered model input. State IDs are comparable only within the same trial.",
    ]:
        y = paragraph(note, y)
    paragraph(
        "Snapshot: "
        + cases["snapshot"]
        + " | Pipeline: "
        + json.dumps(coverage.get("pipeline", {}).get("stages", {})),
        y,
        8,
    )
    footer(2)
    pdf.showPage()
    for page, (group, path, checksum) in enumerate(images, start=3):
        key = f"figure-{page}"
        pdf.bookmarkPage(key)
        pdf.addOutlineEntry(f"{group} / {path.stem}", key)
        pdf.setFillColor(blue)
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(32, height - 33, group + " / " + path.stem)
        reader = ImageReader(str(path))
        iw, ih = reader.getSize()
        scale = min((width - 64) / iw, (height - 97) / ih)
        w, h = iw * scale, ih * scale
        pdf.drawImage(
            reader,
            (width - w) / 2,
            40 + (height - 97 - h) / 2,
            width=w,
            height=h,
            mask="auto",
        )
        footer(page)
        pdf.showPage()
    pdf.save()
    record = dict(
        pdf_sha256=sha(output),
        coverage_sha256=sha(root / "coverage.json"),
        script_sha256=sha(Path(__file__)),
        pages=total,
        inputs=[
            dict(group=g, file=str(p.relative_to(root)), sha256=h) for g, p, h in images
        ],
    )
    output.with_suffix(".sources.json").write_text(json.dumps(record, indent=2))
    print(
        json.dumps(
            dict(path=str(output.resolve()), pages=total, bytes=output.stat().st_size)
        )
    )


if __name__ == "__main__":
    main()
