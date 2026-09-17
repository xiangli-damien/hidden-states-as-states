"""Compare paired final pre/post-RMS means without fitting or changing trials."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hss.data import prepare
from hss.experiments.artifacts import file_digest, save_json
from hss.experiments.config import load
from hss.viz import plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    if cfg.data.representation != "mean":
        raise ValueError("This paired diagnostic expects response-mean features")
    post = prepare(replace(cfg.data, final_norm="post"), cfg.execution.cache_root)
    last = post.layers()[-1]
    if last != post.info["model"][3] - 1:
        raise ValueError("The final captured position must be the model final layer")
    previous = last - 1
    if previous not in post.layers():
        raise ValueError("The previous decoder layer is needed for comparison")
    pre = prepare(
        replace(cfg.data, final_norm="pre", layers=[last]), cfg.execution.cache_root
    )
    if not np.array_equal(post.meta.sample_id, pre.meta.sample_id):
        raise ValueError("Pre/post sample order mismatch")
    arrays = {
        "previous_block": post.array(previous),
        "final_pre_rms": pre.array(last),
        "final_post_rms": post.array(last),
    }
    frame = pd.DataFrame({"sample_id": post.meta.sample_id})
    unit = {}
    for name, values in arrays.items():
        values = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        frame[name + "_norm"] = norms
        unit[name] = values / np.maximum(norms[:, None], 1e-30)
    for a, b in [
        ("previous_block", "final_pre_rms"),
        ("previous_block", "final_post_rms"),
        ("final_pre_rms", "final_post_rms"),
    ]:
        frame[a + "_vs_" + b] = np.einsum("ij,ij->i", unit[a], unit[b])
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "paired_norm_geometry.csv", index=False)
    description = frame.drop(columns="sample_id").describe(
        percentiles=[0.05, 0.5, 0.95]
    )
    description.to_csv(destination / "summary.csv")
    save_json(
        destination / "provenance.json",
        dict(
            pre_snapshot=pre.info["key"],
            post_snapshot=post.info["key"],
            n=len(frame),
            previous_layer=previous,
            final_layer=last,
            config=cfg.to_dict(),
            scope="Paired response means, no refitting. A mean of normalized tokens is not the normalization of their mean. This does not isolate semantics.",
        ),
    )
    with plots.plt.rc_context(plots.STYLE):
        fig, ax = plots.plt.subplots(figsize=(8, 4), layout="constrained")
        for field in frame:
            if "_vs_" in field:
                ax.hist(
                    frame[field],
                    bins=np.linspace(-1, 1, 101),
                    histtype="step",
                    linewidth=1.5,
                    label=field.replace("_", " "),
                )
        ax.set(xlabel="Same-response cosine similarity", ylabel="Responses")
        ax.legend(fontsize=8)
        fig.savefig(destination / "paired_norm_geometry.png")
        fig.savefig(destination / "paired_norm_geometry.svg")
        plots.plt.close(fig)
    import json

    metadata = json.loads((destination / "provenance.json").read_text())
    metadata["script_sha256"] = file_digest(Path(__file__))
    metadata["outputs"] = {
        name: file_digest(destination / name)
        for name in (
            "paired_norm_geometry.csv",
            "summary.csv",
            "paired_norm_geometry.png",
            "paired_norm_geometry.svg",
        )
    }
    save_json(destination / "provenance.json", metadata)
    print(description.to_string())


if __name__ == "__main__":
    main()
