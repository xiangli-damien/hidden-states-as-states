"""Bounded, real-OpenAct integration benchmark. Does not launch generation."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import zarr
from threadpoolctl import threadpool_limits

from hss.experiments.artifacts import save_json, runtime_versions, source_version
from hss.experiments.config import Experiment, ClusterConfig, ExecutionConfig
from hss.experiments.openact import DataSpec, prepare, discover
from hss.experiments.runner import run_experiment
from hss.experiments.sweep import run_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--cuda",
        action="store_true",
        help="Also verify bounded real-data MFA CUDA parity",
    )
    args = p.parse_args()
    spec = DataSpec(
        [args.source],
        layers=[0, -1],
        max_shards=1,
        max_samples=100,
        expected_samples=100,
    )
    result = {
        "runtime": runtime_versions(),
        "source_version": source_version(),
        "source": args.source,
        "checks": {},
    }

    def timed(name, fn):
        start = time.perf_counter()
        value = fn()
        result[name + "_seconds"] = time.perf_counter() - start
        return value

    with threadpool_limits(limits=1):
        cached = timed("prepare_first", lambda: prepare(spec, args.cache))
        timed("prepare_reuse", lambda: prepare(spec, args.cache))
        z = zarr.open_group(str(discover(spec)[0] / "tensors.zarr"), mode="r")
        for layer in cached.layers():
            np.testing.assert_array_equal(
                cached.array(layer), z["hidden_states/mean"][:, layer, :]
            )
        result["checks"]["mean_exact"] = True
        last = cached.layers()[-1]
        for mode in ("mean", "prompt_last"):
            pre = timed(
                "pre_" + mode,
                lambda: prepare(
                    replace(spec, representation=mode, final_norm="pre"), args.cache
                ),
            )
            np.testing.assert_array_equal(
                pre.array(last), z[f"final_norm/pre/{mode}"][:]
            )
        result["checks"]["pre_rms_exact"] = True
        prefix = timed(
            "prefix",
            lambda: prepare(
                replace(
                    spec, representation="prefix", max_samples=2, expected_samples=2
                ),
                args.cache,
            ),
        )
        ptr = z["tokens/sample_ptr"][:]
        for group, rows in prefix.meta.groupby("group_id"):
            for idx, row in rows.iterrows():
                for layer in prefix.layers():
                    expected = np.asarray(
                        z["hidden_states/per_token"][
                            int(ptr[group]) : int(ptr[group]) + int(row.token_end),
                            layer,
                            :,
                        ],
                        dtype="f8",
                    ).mean(0)
                    np.testing.assert_allclose(
                        prefix.array(layer)[idx], expected, atol=1e-6, rtol=1e-6
                    )
        result["checks"]["prefix_exact"] = True
        cfg = Experiment(
            name="real_openact_smoke",
            data=spec,
            cluster=ClusterConfig(k_values=[2, 3], n_init=1, max_iter=5),
            execution=ExecutionConfig(
                cache_root=args.cache,
                output_root=args.output,
                workers=2,
                threads_per_worker=1,
                memory_gib=4,
                min_available_gib=16,
            ),
            grid={"alignment.threshold": [0.4, 0.6]},
        )
        first = timed("sweep_first", lambda: run_sweep(cfg))
        assert first["status"] == "complete", first
        second = timed("sweep_reuse", lambda: run_sweep(cfg))
        assert all(r["cache_hit"] for r in second["results"])
        result["checks"]["sweep_and_resume"] = True
        mfa = replace(
            cfg,
            cluster=ClusterConfig(method="mfa", k=2, rank=2, n_init=1, max_iter=5),
            grid={},
        )
        fitted = timed("mfa_raw", lambda: run_experiment(mfa))
        result["mfa_profile"] = fitted["profile"]
        if args.cuda:
            import torch
            from hss.experiments.runner import _load_layer

            cuda = replace(
                mfa, cluster=replace(mfa.cluster, backend="gpu", device="cuda:0")
            )
            torch.cuda.reset_peak_memory_stats()
            gpu_result = timed("mfa_raw_cuda", lambda: run_experiment(cuda))
            for layer in cached.layers():
                a, pa, _ = _load_layer(fitted["path"], layer)
                b, pb, _ = _load_layer(gpu_result["path"], layer)
                np.testing.assert_allclose(
                    a.score_samples(pa.transform(cached.array(layer))),
                    b.score_samples(pb.transform(cached.array(layer))),
                    rtol=1e-4,
                    atol=1e-3,
                )
            result["cuda_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )
            result["checks"]["real_mfa_cuda_parity"] = True
        result["cache_bytes"] = cached.info["bytes"]
        result["rows"] = cached.n_items()
        result["layers"] = cached.layers()
        result["hidden_dim"] = cached.state_dim()
        result["trials"] = [r["path"] for r in first["results"]] + [fitted["path"]]
    save_json(Path(args.output) / "benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
