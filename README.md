# Hidden States as States

Layer-wise state maps, Mixture of Factor Analyzers (MFA), and reproducible experiment grids over **OpenAct** activations. Analysis does not load language-model weights or run generation.

## Start here

```bash
uv sync --locked --extra dev
uv run hss --help
uv run hss plan configs/smoke.toml
uv run hss sweep configs/smoke.toml
uv run hss report /lambda/nfs/dami/hss-results
```

The shipped paths target Lambda. Override them for another machine:

```bash
uv run hss sweep configs/smoke.toml \
  --set 'data.paths=["/path/to/openact/model/shards"]' \
  --set 'execution.cache_root="/fast/local/hss-cache"' \
  --set 'execution.artifact_cache_root="/persistent/hss-fit-cache"' \
  --set 'execution.output_root="/persistent/hss-results"'
```

`configs/base.toml` requires all 5,000 MATH samples. `smoke.toml` explicitly selects 100. An incomplete full dataset fails before fitting; it is never silently presented as a full run.

## Workflows

| Command | Purpose |
|---|---|
| `hss prepare CONFIG` | Validate published shards; create per-layer local mmap arrays |
| `hss plan CONFIG` | Freeze input snapshot; enumerate trials and estimated RAM |
| `hss run CONFIG` | Run one configuration (grid axes are used by `sweep`) |
| `hss sweep CONFIG` | Bounded process grid, fit reuse, checkpoints, resume |
| `hss paper …` | Generate manuscript experiment configs and dependency manifest |
| `hss suite suite.json --only NAME …` | Execute selected experiments and prerequisites |
| `hss report OUTPUT` | Produce CSV tables and PNG state maps, profiles, ICL surfaces |

```bash
# MFA: low-rank covariance + component-specific diagonal noise.
uv run hss sweep configs/mfa_grid.toml

# Prompt-last prediction: map, PCA, scaling and classifiers fit on train only.
uv run hss run configs/prediction.toml

# Switch the FINAL decoder layer between before/after final RMSNorm.
uv run hss run configs/base.toml --set 'data.final_norm="pre"'

# Optional CUDA mixture fitting; CPU remains the default.
uv sync --locked --extra gpu --extra dev
uv run --extra gpu hss run configs/gpu.toml
```

### Paper experiment suite

```bash
uv run hss paper \
  --data-root /lambda/nfs/dami/openact/runs \
  --directory configs/generated-paper \
  --cache-root /home/ubuntu/hss-cache \
  --output-root /lambda/nfs/dami/hss-results \
  --check-data
uv run hss suite configs/generated-paper/suite.json --only default_map
```

The suite covers geometry, seed/subsample/K-range reliability, fixed-K center comparisons, cross-model/dataset profiles, before-generation prediction, sentence monitoring, characterization, construction controls, and MFA/RMS extensions. Dataset locations are explicit editable configs. Generating a suite does not start experiments or collect missing data.

**A runnable protocol is not evidence that every reported paper number has been reproduced.** Full data collection is ongoing. The supplied manuscript and user-selected dataset sizes differ, safety benchmarks differ from the existing collection, and output-only prefix baselines need additional metrics. See the [reproduction contract](docs/reproduction.md).

## Organization

```text
src/hss/
  cluster/               GMM, KMeans, MFA; numerical models and portable arrays
  align.py, predict.py   Reusable core interfaces
  experiments/
    openact.py           Published-shard reader and local mmap cache
    config.py            Strict TOML/JSON, inheritance, dotted overrides
    fitting.py           Train-only transforms and reusable candidate fits
    runner.py            Split → fit → freeze → evaluate → save
    sweep.py             Bounded process scheduling and frozen plans
    evaluate.py          Categorical NB, holdouts, monitoring and characterization
    diagnostics.py       Separation, dispersion, centroid matching
    paper.py             Experiment catalog and dependencies
    reporting.py         Tables and scientific figures
configs/                 Small editable TOML entry points
scripts/                 Bounded real-data benchmark
tests/                  Numerical, protocol, cache, grid and optional CUDA tests
```

The root `experiments/`, `hss_workbench/`, and `hss_bundle_bridge/` remain available for old notebooks. New work should use `hss.experiments` / `hss`. Legacy prediction accepts an already-fitted map and cannot establish training isolation; it now warns instead of silently presenting that path as the supported reproduction protocol.

- [Configuration and extension guide](docs/configuration.md)
- [Paper methods, coverage and unresolved conditions](docs/reproduction.md)
- [Lambda installation, performance, storage and large grids](docs/lambda.md)
- [MFA implementation](docs/mfa.md)

## Validation

```bash
uv run --extra dev pytest -q
# CUDA parity test is skipped without torch/CUDA.
uv run --extra dev --extra gpu pytest tests/test_gpu_backends.py -q
```

Tests cover dense-covariance/MFA likelihood equivalence, EM behavior, rank-zero diagonal mixtures, OpenAct labels and pre/post RMS, causal prefixes, train/test isolation, response-level FAR calibration, ICL sign, multiprocessing, cache reuse, frozen plans, failure reporting and rendered artifacts.
