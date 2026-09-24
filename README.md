# Hidden States as States

Layer-wise state maps, Mixture of Factor Analyzers (MFA), and reproducible experiment grids over **OpenAct** activations. Core clustering and saved-result analysis do not load language-model weights. Separate, explicitly launched research scripts run model readouts and activation-replacement experiments.

## Start here

[中文使用指南](docs/quickstart.zh-CN.md) · [Architecture and reusable APIs](docs/architecture.md) · [Every paper figure/table and control](docs/paper-artifacts.md) · [Lambda workspace guide](docs/lambda.md)

```bash
uv sync --locked --extra dev
uv run hss --help
uv run hss plan configs/smoke.toml
uv run hss sweep configs/smoke.toml
uv run hss report /lambda/nfs/dami/hss/results/trials
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
| `hss data CATALOG` | Inspect logical dataset paths and published sample counts |
| `hss methods` | List clustering methods/backends and selection criteria |
| `hss results ROOT --validate --full` | Audit saved artifacts, checksums and response splits |
| `hss report OUTPUT` | Aggregate tables and rebuild per-trial PNG/SVG diagnostics |
| `hss figures ROOT --suite SUITE --destination DEST` | Rebuild Figures 1–12/Tables 1–3; explicit missing-input coverage |
| `hss controls ROOT --suite SUITE --destination DEST` | Rebuild supplemental robustness plots/tables; optional bootstrap |

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
  --catalog configs/lambda/datasets.toml \
  --directory /lambda/nfs/dami/hss/studies/paper \
  --cache-root /home/ubuntu/hss-cache \
  --output-root /lambda/nfs/dami/hss/results/trials \
  --check-data
uv run hss suite /lambda/nfs/dami/hss/studies/paper/suite.json --only default_map
```

The suite covers geometry, seed/subsample/K-range reliability, fixed-K center comparisons, cross-model/dataset profiles, before-generation prediction, sentence monitoring, characterization, construction controls, and MFA/RMS extensions. Dataset locations are explicit editable configs. Generating a suite does not start experiments or collect missing data.

**A runnable protocol is not evidence that every reported paper number has been reproduced.** Data and metric coverage vary by study; the supplied manuscript and user-selected dataset sizes and safety benchmarks differ. Consult the dated [execution state](docs/revision-execution-state-20260923.md) for completed work and the [reproduction contract](docs/reproduction.md) for protocol differences.

### Research studies and evidence

The `scripts/` and `configs/` directories also contain separately versioned studies of cluster geometry, state routes, token/layer changes, reconstruction, cross-dataset transfer and activation replacement. They preserve source hashes, frozen selections, original outputs and independent statistical audits. Dated plans are historical protocols, not instructions to restart completed runs.

- [Research execution plan and experiment boundaries](docs/revision-execution-plan-20260923.zh-CN.md)
- [FA/MFA versus PCA replacement: measured output fidelity](docs/revision-fair-factors-results-20260923.zh-CN.md)
- [Completed full GSM8K test collections](docs/gsm8k-test-collection-complete-20260923.zh-CN.md)
- [Steering confirmation and scoring sensitivity](docs/projection-first-v2/scoring-reviewed-sensitivity-20260924.zh-CN.md)

Reconstruction fidelity, answer correctness and causal interpretation are separate endpoints. The current independent steering test has not established a net correctness benefit; partial AI answer review is not full human adjudication. Raw activations, model weights and full answer/report artifacts remain outside Git. The source archive includes study scripts/configurations and the wheel includes the report templates.

## Organization

```text
src/hss/
  data/                  OpenAct + array adapters, dataset catalog, local mmap cache
  cluster/               GMM, MFA, KMeans/MiniBatchKMeans; portable model arrays
  transform/             Train-only preprocessing and portable projections
  results/               Lazy result/model reading and checksum/structure audit
  analysis/              Numerical tables, Hamming geometry, comparisons/bootstrap
  route/                 Label-free route counts, latent chains, neural models, inference
  viz/                   Pure plots, paper recipes, control reports, render manifests
  align.py, predict.py   Reusable core interfaces
  experiments/
    openact.py           Compatibility import for hss.data
    config.py            Strict TOML/JSON, inheritance, dotted overrides
    fitting.py           Train-only transforms and reusable candidate fits
    runner.py            Split → fit → freeze → evaluate → save
    sweep.py             Bounded process scheduling and frozen plans
    evaluate.py          Categorical NB, holdouts, monitoring and characterization
    diagnostics.py       Separation, dispersion, centroid matching
    paper.py             Experiment catalog and dependencies
    reporting.py         Compatibility entry for independent table/plot modules
configs/                 Small editable TOML entry points
scripts/                 Bounded real-data benchmark
tests/                  Numerical, protocol, cache, grid and optional CUDA tests
```

The root `experiments/`, `hss_workbench/`, and `hss_bundle_bridge/` remain available for old notebooks. New work should use `hss.experiments` / `hss`. Legacy prediction accepts an already-fitted map and cannot establish training isolation; it now warns instead of silently presenting that path as the supported reproduction protocol.

- [Configuration and extension guide](docs/configuration.md)
- [Paper methods, coverage and unresolved conditions](docs/reproduction.md)
- [Lambda installation, performance, storage and large grids](docs/lambda.md)
- [MFA implementation](docs/mfa.md)
- [Residual-channel correctness study: held-out effects, transfer and temporal checks](docs/channel-study.zh-CN.md)
- [Confidence scalars, full-dimensional directions, and cross-layer follow-up](docs/confidence-study.zh-CN.md)
- [Confidence follow-up findings and limits](docs/confidence-findings.zh-CN.md)
- [State-route research protocol](docs/state-route-research-plan.zh-CN.md)
- [Initial route findings and structural controls](docs/state-route-initial-results.zh-CN.md)
- [Unsupervised route algorithm benchmark and new-sample scoring](docs/route-algorithm-benchmark.zh-CN.md)
- [GMMs of cross-layer updates and adjacent-token differences](docs/change-clusters.zh-CN.md)

## Plot without refitting

```bash
hss figures /lambda/nfs/dami/hss/results/trials \
  --suite /lambda/nfs/dami/hss/studies/paper/suite.json \
  --destination /lambda/nfs/dami/hss/figures/paper \
  --only figure_03 figure_04 --max-trajectories 5000
```

Open `figures/paper/index.html` for the gallery. Each figure includes its source CSV/NPY data and a manifest of trial IDs, parameters and hashes. Results can move machines and render without raw activations or fit caches. Plot edits do not invalidate fitting checkpoints. Full coverage and unresolved paper conditions are documented in the [artifact inventory](docs/paper-artifacts.md).

## Validation

The Llama–MATH review also provides a manuscript-ordered `paper.html`, separate
method galleries, and an original-response viewer. [Figure design and reference
code audit](docs/figure-design.zh-CN.md) explains the visual encodings and remaining
protocol differences. Set `HSS_FIGURE_STYLE=configs/figures.toml` to adjust rendering
without refitting; resolved settings and displayed-edge masks are recorded.

[Actual Lambda validation and measured timings](docs/validation.md).

```bash
uv run --extra dev pytest -q
# CUDA parity test is skipped without torch/CUDA.
uv run --extra dev --extra gpu pytest tests/test_gpu_backends.py -q
```

Tests cover dense-covariance/MFA likelihood equivalence, EM behavior, rank-zero diagonal mixtures, OpenAct labels and pre/post RMS, causal prefixes, train/test isolation, response-level FAR calibration, ICL sign, multiprocessing, cache reuse, frozen plans, failure reporting and rendered artifacts.
