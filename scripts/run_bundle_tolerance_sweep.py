#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.config import DiscretizeConfig, PredictionConfig, SelectConfig
from experiments.prediction import run_prediction
from hss_workbench import Workspace


def _parse_tolerances(args: argparse.Namespace) -> np.ndarray:
    if args.tolerances:
        return np.array([float(x) for x in args.tolerances], dtype=np.float64)
    lo = float(args.tol_min)
    hi = float(args.tol_max)
    n = int(args.tol_points)
    if n < 2:
        return np.array([lo], dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _tol_slug(tol: float) -> str:
    s = f"{tol:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _k_stats(k_map: dict[int, int]) -> dict[str, float | int]:
    if not k_map:
        return {"n_layers": 0, "k_mean": 0.0, "k_std": 0.0, "k_min": 0, "k_max": 0}
    ks = np.array(list(k_map.values()), dtype=np.float64)
    return {
        "n_layers": int(len(ks)),
        "k_mean": float(np.mean(ks)),
        "k_std": float(np.std(ks)),
        "k_min": int(np.min(ks)),
        "k_max": int(np.max(ks)),
    }


def _load_prediction_config(ws: Workspace, name: Optional[str], fallback: PredictionConfig) -> PredictionConfig:
    if not name:
        return fallback
    art = ws.store.load_manifest("analysis", name)
    cfg = dict(art.config or {})
    if not cfg and ws.analysis_store(name).exists("prediction/prediction_config.json"):
        cfg = ws.analysis_store(name).load_json("prediction/prediction_config.json")
    return PredictionConfig.from_dict(cfg) if cfg else fallback


def _run_progressive_prefix(
    *,
    ws: Workspace,
    discretize_name: str,
    prediction_cfg: PredictionConfig,
    prefix_step: int,
) -> pd.DataFrame:
    disc = ws.load_discretize(discretize_name)
    data = ws.open_data()
    gl = np.asarray(disc.global_labels)
    layers = list(disc.layers)
    y = np.asarray(data.y)
    if gl.ndim != 2:
        raise ValueError(f"Expected 2-D global_labels, got shape={gl.shape}")
    rows = []
    prefixes = list(range(max(1, prefix_step), len(layers) + 1, max(1, prefix_step)))
    if prefixes[-1] != len(layers):
        prefixes.append(len(layers))
    for prefix_len in prefixes:
        gl_prefix = gl[:, :prefix_len]
        layers_prefix = layers[:prefix_len]
        pred = run_prediction(data.provider, gl_prefix, y, layers_prefix, prediction_cfg, store=None)
        for _, row in pred.summary_df.iterrows():
            rows.append(
                {
                    "prefix_len": int(prefix_len),
                    "last_layer": int(layers_prefix[-1]),
                    "method": str(row["method"]),
                    "mean_auroc": float(row.get("mean_auroc", np.nan)),
                    "std_auroc": float(row.get("std_auroc", np.nan)),
                    "mean_accuracy": float(row.get("mean_accuracy", np.nan)),
                    "std_accuracy": float(row.get("std_accuracy", np.nan)),
                    "mean_f1": float(row.get("mean_f1", np.nan)),
                    "std_f1": float(row.get("std_f1", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Run per-tolerance select/discretize/prediction sweeps inside an existing HSS workspace.")
    p.add_argument("--workspace-root", required=True)
    p.add_argument("--scan-name", default="scan")
    p.add_argument("--base-discretize-name", default="discretize")
    p.add_argument("--base-prediction-name", default=None)
    p.add_argument("--tolerances", nargs="*", default=None)
    p.add_argument("--tol-min", type=float, default=0.00)
    p.add_argument("--tol-max", type=float, default=0.10)
    p.add_argument("--tol-points", type=int, default=11)
    p.add_argument("--stability-min", type=float, default=0.30)
    p.add_argument("--run-prediction", action="store_true")
    p.add_argument("--prediction-methods", nargs="*", default=None)
    p.add_argument("--prediction-splits", type=int, default=5)
    p.add_argument("--prediction-seed", type=int, default=42)
    p.add_argument("--prefix-step", type=int, default=0, help="If >0, run progressive prefix prediction every N layers using each discretize artifact.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ws_root = Path(args.workspace_root).expanduser().resolve()
    ws = Workspace(ws_root, quiet=True)
    _ = ws.open_data()
    _ = ws.load_scan(args.scan_name)
    base_disc = ws.load_discretize(args.base_discretize_name)
    base_disc_cfg = DiscretizeConfig.from_dict(base_disc.config)

    default_pred_cfg = PredictionConfig(
        methods=args.prediction_methods or ["HSS-NB", "HSS-Markov", "GaussianNB", "Logistic"],
        n_splits=int(args.prediction_splits),
        seed=int(args.prediction_seed),
    )
    pred_cfg = _load_prediction_config(ws, args.base_prediction_name, default_pred_cfg)
    if args.prediction_methods:
        pred_cfg = replace(pred_cfg, methods=list(args.prediction_methods))
    pred_cfg = replace(pred_cfg, n_splits=int(args.prediction_splits), seed=int(args.prediction_seed))

    tolerances = _parse_tolerances(args)
    out_root = ws_root / "sweeps"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    prediction_rows = []
    progressive_rows = []

    for tol in tolerances:
        tol = float(tol)
        slug = _tol_slug(tol)
        select_name = f"select_tol_{slug}"
        disc_name = f"discretize_tol_{slug}"
        pred_name = f"prediction_tol_{slug}"

        if args.overwrite or not ws.store.has_artifact("select", select_name):
            select_cfg = SelectConfig(
                strategy="icl_parsimonious",
                relative_pct=tol,
                stability_min=float(args.stability_min),
            )
            ws.run_select(select_name, scan=args.scan_name, config=select_cfg, overwrite=args.overwrite)
        select_art = ws.load_select(select_name)

        if args.overwrite or not ws.store.has_artifact("discretize", disc_name):
            ws.run_discretize(disc_name, select=select_name, config=base_disc_cfg, overwrite=args.overwrite)
        disc_art = ws.load_discretize(disc_name)

        k_stats = _k_stats(select_art.k_map)
        manifest_rows.append(
            {
                "tolerance": tol,
                "select_name": select_name,
                "discretize_name": disc_name,
                "prediction_name": pred_name if args.run_prediction else None,
                "n_global_states": int(disc_art.n_global_states),
                **k_stats,
            }
        )

        if args.run_prediction:
            if args.overwrite or not ws.store.has_artifact("analysis", pred_name):
                ws.run_prediction(pred_name, discretize=disc_name, config=pred_cfg, overwrite=args.overwrite)
            pred = ws.load_prediction(pred_name)
            for _, row in pred.summary_df.iterrows():
                prediction_rows.append(
                    {
                        "tolerance": tol,
                        "prediction_name": pred_name,
                        "method": str(row["method"]),
                        "mean_auroc": float(row.get("mean_auroc", np.nan)),
                        "std_auroc": float(row.get("std_auroc", np.nan)),
                        "mean_accuracy": float(row.get("mean_accuracy", np.nan)),
                        "std_accuracy": float(row.get("std_accuracy", np.nan)),
                        "mean_f1": float(row.get("mean_f1", np.nan)),
                        "std_f1": float(row.get("std_f1", np.nan)),
                    }
                )

        if int(args.prefix_step) > 0:
            prefix_df = _run_progressive_prefix(
                ws=ws,
                discretize_name=disc_name,
                prediction_cfg=pred_cfg,
                prefix_step=int(args.prefix_step),
            )
            if len(prefix_df) > 0:
                prefix_df = prefix_df.copy()
                prefix_df.insert(0, "tolerance", tol)
                progressive_rows.extend(prefix_df.to_dict(orient="records"))
                prefix_path = out_root / f"progressive_prefix_tol_{slug}.csv"
                prefix_df.to_csv(prefix_path, index=False)

    manifest_df = pd.DataFrame(manifest_rows).sort_values("tolerance").reset_index(drop=True)
    manifest_df.to_csv(out_root / "tolerance_manifest.csv", index=False)
    (out_root / "tolerance_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows).sort_values(["tolerance", "method"]).reset_index(drop=True)
        pred_df.to_csv(out_root / "tolerance_prediction_summary.csv", index=False)

    if progressive_rows:
        prog_df = pd.DataFrame(progressive_rows).sort_values(["tolerance", "prefix_len", "method"]).reset_index(drop=True)
        prog_df.to_csv(out_root / "progressive_prefix_summary.csv", index=False)

    print("=" * 80)
    print("Tolerance sweep complete")
    print("=" * 80)
    print(f"workspace_root: {ws_root}")
    print(f"base_scan: {args.scan_name}")
    print(f"base_discretize: {args.base_discretize_name}")
    print(f"tolerances: {[float(x) for x in tolerances.tolist()]}")
    print(f"outputs: {out_root}")
    print("\nArtifacts per tolerance:")
    for row in manifest_rows:
        print(
            f"  tol={row['tolerance']:.4f} -> {row['select_name']} | {row['discretize_name']}"
            + (f" | {row['prediction_name']}" if row.get("prediction_name") else "")
        )


if __name__ == "__main__":
    main()
