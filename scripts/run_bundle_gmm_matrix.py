#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.config import DiscretizeConfig, PredictionConfig, ScanConfig, SelectConfig
from experiments.io_utils import ExperimentStore
from hss_bundle_bridge import BundleAdapter, SelectionSpec
from hss_workbench import Workspace


@dataclass(frozen=True)
class ComboSpec:
    view_name: str
    preprocess_name: str
    transform_steps: list[dict[str, Any]]


def parse_selection(args: argparse.Namespace) -> Optional[SelectionSpec]:
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x) for x in args.sample_ids.split(",") if str(x).strip()]
    if not any([sample_ids, args.query, args.head is not None, args.tail is not None, args.frac is not None]):
        return None
    return SelectionSpec(
        sample_ids=sample_ids,
        query=args.query,
        head=args.head,
        tail=args.tail,
        frac=args.frac,
        random_state=args.seed,
    )


def parse_layers(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(x) for x in raw.split(",") if str(x).strip()]


def summarize_k_map(k_map: dict[int, int]) -> dict[str, Any]:
    if not k_map:
        return {"n_layers": 0, "k_mean": None, "k_std": None, "k_min": None, "k_max": None}
    ks = np.array(list(k_map.values()), dtype=np.float64)
    return {
        "n_layers": int(len(ks)),
        "k_mean": float(np.mean(ks)),
        "k_std": float(np.std(ks)),
        "k_min": int(np.min(ks)),
        "k_max": int(np.max(ks)),
    }


def maybe_run_scan(ws: Workspace, name: str, *, cfg: ScanConfig, layers: Optional[list[int]], overwrite: bool):
    if not overwrite and ws.store.has_artifact("scan", name):
        return ws.load_scan(name)
    return ws.run_scan(name, config=cfg, layers=layers, overwrite=overwrite)


def maybe_run_select(ws: Workspace, name: str, *, scan_name: str, cfg: SelectConfig, overwrite: bool):
    if not overwrite and ws.store.has_artifact("select", name):
        return ws.load_select(name)
    return ws.run_select(name, scan=scan_name, config=cfg, overwrite=overwrite)


def maybe_run_discretize(ws: Workspace, name: str, *, select_name: str, cfg: DiscretizeConfig, layers: Optional[list[int]], overwrite: bool):
    if not overwrite and ws.store.has_artifact("discretize", name):
        return ws.load_discretize(name)
    return ws.run_discretize(name, select=select_name, config=cfg, layers=layers, overwrite=overwrite)


def maybe_run_prediction(ws: Workspace, name: str, *, discretize_name: str, cfg: PredictionConfig, overwrite: bool):
    if not overwrite and ws.store.has_artifact("analysis", name):
        return ws.load_prediction(name)
    ws.run_prediction(name, discretize=discretize_name, config=cfg, overwrite=overwrite)
    return ws.load_prediction(name)


def combo_output_dir(output_root: Path, spec: ComboSpec) -> Path:
    return output_root / f"{spec.view_name}__{spec.preprocess_name}__gmm"


def persist_and_reopen_workspace(outdir: Path, view_loaded_inputs, *, quiet: bool, log_level: str) -> Workspace:
    ws = Workspace(outdir, quiet=quiet, log_level=log_level, thread_limit=1)
    ws.load_data(view_loaded_inputs, persist=True)
    return Workspace(outdir, quiet=quiet, log_level=log_level, thread_limit=1)


def run_one_combo(
    *,
    bundle_dir: Path,
    output_root: Path,
    selection: Optional[SelectionSpec],
    layers: Optional[list[int]],
    spec: ComboSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    adapter = BundleAdapter(bundle_dir)
    available_views = set(adapter.list_views())
    if spec.view_name not in available_views:
        return {
            "status": "skipped",
            "reason": f"missing view {spec.view_name}",
            "view_name": spec.view_name,
            "preprocess_name": spec.preprocess_name,
        }

    outdir = combo_output_dir(output_root, spec)
    outdir.mkdir(parents=True, exist_ok=True)

    view = adapter.open_view(spec.view_name, selection=selection, layers=layers)
    ws = persist_and_reopen_workspace(outdir, view.to_experiments_loaded_inputs(), quiet=args.quiet, log_level=args.log_level)

    scan_cfg = ScanConfig(
        k_range=(args.k_min, args.k_max),
        seed=args.seed,
        stability_repeats=args.stability_repeats,
        eval_sample=min(args.eval_sample, view.n_items),
        icl_mode=args.icl_mode,
        method="gmm",
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        batch_size=args.scan_batch_size,
        n_jobs=args.scan_n_jobs,
        parallel_backend=args.scan_parallel_backend,
        transform_steps=spec.transform_steps,
    )
    select_cfg = SelectConfig(
        strategy="icl_parsimonious",
        relative_pct=args.relative_pct,
        stability_min=args.stability_min,
    )
    disc_cfg = DiscretizeConfig(
        seed=args.seed,
        method="gmm",
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        transform_steps=spec.transform_steps,
        alignment_similarity=args.alignment_similarity,
        alignment_method=args.alignment_method,
        alignment_threshold=args.alignment_threshold,
        batch_size_fit=args.fit_batch_size,
        batch_size_predict=args.predict_batch_size,
        n_jobs=args.discretize_n_jobs,
        parallel_backend=args.discretize_parallel_backend,
        parsimony_tolerance=args.relative_pct,
        icl_mode=args.icl_mode,
    )

    scan_art = maybe_run_scan(ws, "scan", cfg=scan_cfg, layers=layers, overwrite=args.overwrite)
    select_art = maybe_run_select(ws, "select", scan_name="scan", cfg=select_cfg, overwrite=args.overwrite)
    disc_art = maybe_run_discretize(ws, "discretize", select_name="select", cfg=disc_cfg, layers=layers, overwrite=args.overwrite)

    prediction_summary: list[dict[str, Any]] = []
    prediction_methods: list[str] = []
    if args.run_prediction and view.y is not None and len(view.y) > 0 and np.unique(view.y).shape[0] >= 2:
        methods = [m.strip() for m in args.prediction_methods.split(",") if m.strip()]
        pred_cfg = PredictionConfig(
            n_splits=args.prediction_splits,
            seed=args.seed,
            alpha=1.0,
            methods=methods,
            continuous_feature_mode=args.continuous_feature_mode,
            continuous_standardize=not args.no_continuous_standardize,
        )
        pred = maybe_run_prediction(ws, "prediction", discretize_name="discretize", cfg=pred_cfg, overwrite=args.overwrite)
        prediction_methods = methods
        prediction_summary = pred.summary_df.to_dict(orient="records")

    root_store = ExperimentStore(outdir)
    run_spec = {
        "bundle_dir": str(bundle_dir),
        "selection": None if selection is None else selection.to_dict(),
        "requested_layers": layers,
        "view_name": spec.view_name,
        "view_unit": view.view_manifest.unit,
        "preprocess_name": spec.preprocess_name,
        "transform_steps": spec.transform_steps,
        "scan_config": scan_cfg.to_dict(),
        "select_config": select_cfg.to_dict(),
        "discretize_config": disc_cfg.to_dict(),
        "prediction_methods": prediction_methods,
        "source_provider_manifest": view.provider.metadata_manifest(),
    }
    root_store.save_json("meta/run_spec.json", run_spec)

    return {
        "status": "done",
        "workspace_root": str(outdir),
        "view_name": spec.view_name,
        "preprocess_name": spec.preprocess_name,
        "transform_steps": spec.transform_steps,
        "n_items": int(view.n_items),
        "n_layers": int(len(view.layers)),
        "state_dim": int(view.state_dim),
        "positive_rate": float(np.mean(view.y)) if len(view.y) > 0 else None,
        "k_summary": summarize_k_map({int(k): int(v) for k, v in select_art.k_map.items()}),
        "n_global_states": int(disc_art.n_global_states),
        "scan_rows": int(len(scan_art.metrics_df)),
        "prediction_summary": prediction_summary,
    }


def build_specs(views: Iterable[str], preprocess_names: Iterable[str]) -> list[ComboSpec]:
    step_map: dict[str, list[dict[str, Any]]] = {
        "raw": [],
        "standardize": [{"name": "standardize"}],
    }
    specs: list[ComboSpec] = []
    for view_name in views:
        for preprocess_name in preprocess_names:
            if preprocess_name not in step_map:
                raise ValueError(f"unsupported preprocess {preprocess_name!r}")
            specs.append(ComboSpec(view_name=view_name, preprocess_name=preprocess_name, transform_steps=step_map[preprocess_name]))
    return specs


def write_registry(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["transform_steps"] = json.dumps(flat.get("transform_steps"), ensure_ascii=False)
        flat["k_summary"] = json.dumps(flat.get("k_summary"), ensure_ascii=False)
        flat["prediction_summary"] = json.dumps(flat.get("prediction_summary"), ensure_ascii=False)
        flat_rows.append(flat)

    if flat_rows:
        keys = sorted({key for row in flat_rows for key in row.keys()})
        with (output_root / "matrix_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 4-combo GMM artifact matrix on a bundle.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--views", nargs="*", default=["sample_mean", "sample_prompt_last"])
    parser.add_argument("--preprocess", nargs="*", default=["raw", "standardize"])
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer ids. Default: all layers.")
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--tail", type=int, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--stability-repeats", type=int, default=2)
    parser.add_argument("--eval-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--covariance-type", type=str, default="diag")
    parser.add_argument("--reg-covar", type=float, default=1e-2)
    parser.add_argument("--icl-mode", type=str, default="bic_plus_2entropy")
    parser.add_argument("--relative-pct", type=float, default=0.02)
    parser.add_argument("--stability-min", type=float, default=0.30)
    parser.add_argument("--alignment-similarity", type=str, default="cosine")
    parser.add_argument("--alignment-method", type=str, default="hungarian")
    parser.add_argument("--alignment-threshold", type=float, default=0.0)
    parser.add_argument("--scan-batch-size", type=int, default=4096)
    parser.add_argument("--fit-batch-size", type=int, default=4096)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--scan-n-jobs", type=int, default=1)
    parser.add_argument("--discretize-n-jobs", type=int, default=1)
    parser.add_argument("--scan-parallel-backend", type=str, default="loky")
    parser.add_argument("--discretize-parallel-backend", type=str, default="loky")
    parser.add_argument("--run-prediction", action="store_true")
    parser.add_argument("--prediction-methods", type=str, default="HSS-NB,GaussianNB,Logistic")
    parser.add_argument("--prediction-splits", type=int, default=5)
    parser.add_argument("--continuous-feature-mode", type=str, default="last_layer", choices=["last_layer", "all_layers"])
    parser.add_argument("--no-continuous-standardize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    if not args.bundle_dir.exists():
        raise FileNotFoundError(str(args.bundle_dir))

    selection = parse_selection(args)
    layers = parse_layers(args.layers)
    specs = build_specs(args.views, args.preprocess)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[RUN] {spec.view_name} / {spec.preprocess_name}")
        row = run_one_combo(
            bundle_dir=args.bundle_dir,
            output_root=args.output_root,
            selection=selection,
            layers=layers,
            spec=spec,
            args=args,
        )
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

    write_registry(args.output_root, rows)
    print(f"\nSaved registry to {args.output_root / 'matrix_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
