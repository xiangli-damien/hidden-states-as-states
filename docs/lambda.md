# Lambda: installation and efficient execution

Code synchronization follows **local commit → GitHub push → GPU fetch/checkout**. Keep results under `/lambda/nfs/dami` and derived activation caches on the instance's local disk.

```bash
ssh gpu2
cd /lambda/nfs/dami
# First installation:
git clone git@github.com:xiangli-damien/hidden-states-as-states.git
cd hidden-states-as-states
git fetch origin
git switch --track origin/codex/openact-mfa-grid
# Subsequent updates on this branch:
git pull --ff-only
# Create the environment on local disk; leave code/results on dami.
uv venv --python 3.11 /home/ubuntu/hss-venv
ln -s /home/ubuntu/hss-venv .venv
uv sync --locked --extra dev
```

Install `uv` using your normal package manager if it is not on PATH. This package has its own `.venv`; do not upgrade the environment of an active OpenAct collection job. To install CUDA dependencies, use `uv sync --locked --extra dev --extra gpu` and include `--extra gpu` in `uv run` commands so the extra remains selected. The CUDA wheel/driver must be compatible with the host.

## Why loading is faster

Published OpenAct Zarr shards are immutable input. `hss prepare` reads the chosen representation once into contiguous local **one-file-per-layer NPY** arrays. Workers read those arrays by memory mapping, so processes share the operating system's file cache and do not pickle large activation arrays. Mean and prompt-last modes read precomputed aggregates directly; they do not rescan all token activations. Prefix/token modes read token chunks once during preparation.

Example raw float32 aggregate sizes, before small metadata: `samples × layers × hidden_dim × 4`.

| Full MATH (5,000) | Approximate cached aggregate size per representation |
|---|---:|
| Llama-3.2-1B, 17×2048 | 0.65 GiB |
| Qwen2-7B, 29×3584 | 1.94 GiB |
| Llama-3-8B, 33×4096 | 2.52 GiB |

Prefix and full-token arrays can be much larger. The planner checks the expected cache size against free disk plus a reserve. Cache preparation still depends on NFS throughput; warm reuse does not remove the cost of fitting new mixtures.

The plan also reports unique candidate-fit counts and candidate parameter storage before checkpoint reuse. For example, Qwen2 raw MFA with rank 32, K=2…80 and 29 layers stores roughly **85 GiB of candidate parameter arrays per seed**. A multi-rank/multi-seed search can therefore consume hundreds of GiB even when the raw aggregate cache is small. This estimate excludes final exports and filesystem overhead; inspect the storage quota before choosing the full grid.

## Resource limits and resume

- `workers × threads_per_worker` bounds CPU fitting parallelism. Use 2 workers × 2 threads initially while collection runs, then measure.
- The scheduler caps concurrency using estimated per-trial RAM, configured budget, available RAM and CPU count. Estimates are conservative planning estimates, not an OS memory limit.
- CUDA mixture fits share a per-device advisory lock across this package's processes; free-memory checks leave a configured reserve. Other programs, including OpenAct, do not obey this lock. GPU fitting may slow active generation even when memory fits.
- Keep CPU as the default during active collection. Use a small explicit CUDA smoke to verify compatibility; schedule large CUDA grids once GPU capacity is available.
- Completed fits checkpoint **per layer, K and fitting configuration**. Changing eta, report choices or K-selection tolerance reuses compatible transforms/mixtures. Adding K candidates fits only missing candidates. Changing seed/rank/PCA fits the genuinely different model.
- Rerun the same command after interruption. A frozen plan prevents new collecting shards from changing sample membership mid-search. Use `--refresh-data` to adopt new shards explicitly.

```bash
# Review scale and RAM before launching a large grid.
hss plan configs/mfa_grid.toml
# Background execution survives SSH disconnects.
mkdir -p /lambda/nfs/dami/hss-results
tmux new-session -d -s hss-grid 'cd /lambda/nfs/dami/hidden-states-as-states && .venv/bin/hss sweep configs/mfa_grid.toml > /lambda/nfs/dami/hss-results/grid.log 2>&1'
```

`status.json` and `sweeps/*/task_*_status.json` provide machine-readable progress. Failed trials retain errors/tracebacks; sweep exit status is nonzero. No completed result is overwritten by an incomplete retry.

For multiple CPU machines sharing persistent results, give each a local data cache and use stable partitions:

```bash
hss sweep configs/mfa_grid.toml --set 'execution.task_count=4' --set 'execution.task_index=0'
# Use indices 1, 2, 3 on the other workers, with matching configs and cache paths.
```

A saved plan contains absolute paths. On multiple hosts, use the same absolute local-cache path and prepare equivalent frozen snapshots on each host before sharing a plan. NFS locking must be supported; this is not a distributed job service. One Lambda machine's process pool is the tested scheduling target.

## Measure the real workload

```bash
.venv/bin/python scripts/benchmark_openact.py \
  --source /lambda/nfs/dami/openact/runs/math_full_20260916/llama32 \
  --cache /home/ubuntu/hss-benchmark-cache \
  --output /lambda/nfs/dami/hss-benchmark
```

This bounds reads to 100 samples/two layers (two samples for prefixes), checks cached values against source arrays, fits small GMM/MFA models, measures a two-trial sweep, and verifies resume. `benchmark.json` records actual times and versions. It is not a full 5,000-sample, 80-component grid timing estimate. Use those intended dimensions for a separate capacity benchmark before budgeting the full run.

Persistent results include portable model/projection arrays and labels/state sequences. Local derived raw caches can be rebuilt from verified NFS shards after replacing the GPU instance. Preserve generated configs, the frozen plan and benchmark/results with the code revision.

## Current organized workspace

New runs use the following layout. Earlier `hss-benchmark-*` and `hss-experiments` directories are retained as historical artifacts; their data are not silently moved or overwritten.

```text
/lambda/nfs/dami/
  openact/                         Collection code and published runs (unchanged)
  hidden-states-as-states/          HSS Git checkout; .venv → /home/ubuntu/hss-venv
  hss/
    README.md / workspace.json     Paths, code checkout and artifact roles
    studies/paper/                 Resolved configs, suite.json, suite_status.json
    results/trials/                Immutable completed scientific trials
    cache/fits/                    Persistent transform/candidate-fit checkpoints
    figures/paper/                 PNG/SVG/CSV/NPY + manifests + HTML gallery
    figures/controls/              Supplemental controls and uncertainty tables
    reports/                       Audits, availability and coverage manifests
    validation/                    Bounded CPU/synthetic validation, separately named
/home/ubuntu/hss-cache/             Rebuildable local mmap activation cache
```

Dataset paths are configured in `configs/lambda/datasets.toml`, not inside fitting or plotting code. `hss paper --catalog ...` copies resolved mappings into a persistent study. Generate once per study; use a new directory for a changed scientific protocol. Existing studies are protected against regeneration unless `--overwrite` is explicit. Rerun `hss suite ... --only JOB` to resume. Separate job invocations retain each other's status under a file lock.

Changing the study JSON after a job finishes is detected during figure resolution. The original trial still contains its exact configuration; rerun the changed job or use a separately preserved study. Keep the HSS checkout stable while a grid is active; update code before launching a new study.

### Bounded architecture validation

```bash
# Real captures, three CPU clustering methods, prediction, full checksums,
# resume and standalone paper diagnostics; no generation/model weights.
.venv/bin/python scripts/validate_lambda_workflow.py \
  --source /lambda/nfs/dami/openact/runs/math_full_20260916/llama32 \
  --cache /home/ubuntu/hss-cache \
  --directory /lambda/nfs/dami/hss/validation/real-UNIQUE_NAME

# All recipes/controls on explicitly synthetic 40-response, 6-dimensional data.
.venv/bin/python scripts/smoke_reproduction.py \
  --directory /lambda/nfs/dami/hss/validation/synthetic-UNIQUE_NAME
```

These scripts use bounded engineering fixtures, not full paper hyperparameters. Use fresh validation directories. The real script fits only 3 layers, K=2 and 6 iterations with two CPU workers; the synthetic script exercises all figure/control routes. Production collection remains the primary GPU workload.
