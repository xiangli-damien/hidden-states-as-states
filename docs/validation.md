# Validation: Lambda A100, 2026-09-16

## Reorganized workflow validation — 2026-09-17 UTC

Code commit `377063496ee90774485a1f5511da5c51594dc238`; [machine-readable evidence](validation/lambda-architecture-20260917.json). The subsequent documentation commit does not change executable source.

- Local tests: **27 passed, 1 CUDA test skipped**, 15.02 s.
- Lambda with `CUDA_VISIBLE_DEVICES=""`: **27 passed, 1 skipped**, 41.00 s. The production GPU remained assigned to OpenAct.
- Wheel build succeeded and contains the new `data`, `results`, `analysis`, `viz` and assignment modules.
- Real OpenAct: first **100 published Llama-3.2 MATH responses**, layers **0/8/16**, K=2, six fitting iterations, two CPU workers. GMM, MFA and MiniBatchKMeans complete; prompt-last HSS-NB/Logistic prediction, artifact hashes, split checks and standalone report/figure generation pass.
- Synthetic complete workflow: **46 jobs / 69 distinct trials**, Figures **1–12**, Tables **1–3**, and **17 supplemental control groups** rendered on Lambda in **303.64 s**. Every result passes full checksum audit. Synthetic fixtures are clearly separated from scientific results and marked on images.
- Whole-data loading: the completed **5,000-response, 17-layer, 2,048-dimensional** Llama-3.2 MATH generation-mean cache contains **696,320,000 bytes** of float32 arrays (~0.65 GiB). Preparation took **34.11 s**; reopening the identical mmap cache took **0.432 s**. This is an aggregate cache, not the full per-token capture.

| Real-data smoke operation | Measured seconds |
|---|---:|
| Prepare 100-response aggregate cache | 0.732 |
| Reopen the cache | 0.039 |
| Three-method CPU sweep | 4.917 |
| Resume the same sweep (all three cache hits) | 0.456 |
| Prompt-last prediction | 1.614 |
| Full result checksums and structural audit | 0.205 |
| Rebuild reports plus 100-repeat bootstrap | 6.277 |
| Rebuild paper diagnostics | 2.514 |

These timings validate bounded workflows; they do not estimate a full 5,000-sample, 80-component grid. Short iteration caps deliberately prioritize path validation and can emit convergence warnings. Whole-data experiments still require convergence/selection inspection and the missing datasets/metrics listed in [paper coverage](paper-artifacts.md).

Persistent artifacts:

```text
/lambda/nfs/dami/hss/validation/real-20260917/
/lambda/nfs/dami/hss/validation/synthetic-20260917/
/lambda/nfs/dami/hss/validation/full-math-loading-20260917.json
/lambda/nfs/dami/hss/reports/dataset-availability.json
/lambda/nfs/dami/hss/reports/paper-coverage/
```

The source captures and running OpenAct collection code were not modified. New HSS code arrived via local commit → GitHub push → Lambda fast-forward pull.


Implementation commit: `af8581386ac1c66fd567f546a957f0d450a2ad72`, package 0.4.0. Source/runtime fingerprint and detailed timings are in [the machine-readable benchmark](validation/lambda-smoke-20260916.json).

## Results

- **16/16 tests passed on Lambda**, including CPU/CUDA numerical parity for both GMM and MFA; final full test run took 29.14 seconds.
- GitHub CI passed on Python 3.11 and 3.12, including source/wheel builds. [Implementation validation run](https://github.com/xiangli-damien/hidden-states-as-states/actions/runs/35142961528).
- Real OpenAct generation means exactly matched the cached arrays. Pre-final-RMS generation means and prompt-last vectors exactly matched the explicit OpenAct pre-norm arrays. Prefix means matched independently computed causal token means.
- Real-data MFA CPU and CUDA likelihoods agreed within the benchmark's stated tolerance (`rtol=1e-4`, `atol=1e-3`). PyTorch's peak allocated tensor memory during this small CUDA fit was **14.78 MiB**; this excludes CUDA context/driver and other processes' memory.
- Two-trial GMM grid completed, reused candidate fits across eta changes, and resumed without refitting or launching worker processes for completed tasks.
- Reports rendered four actual result sets (GMM eta variants, CPU MFA, CUDA MFA).

## Measured scope and timings

A100-SXM4 **40 GB**, Python 3.11.16, Torch 2.14.0, NumPy 2.4.6; production collection continued on the same GPU. The benchmark used the first published **100 MATH responses from Llama-3.2-1B**, layers 0 and 16, hidden dimension 2048. Prefix checking used two responses. Cache matrices totalled 1,638,400 bytes.

| Operation | Seconds |
|---|---:|
| First derived aggregate-cache build | 0.388 |
| Reuse the same data cache | 0.013 |
| GMM grid: K=2,3; eta=.4,.6; two layers; 5 iterations, one restart | 3.075 |
| Resume the completed grid | 0.125 |
| CPU MFA: K=2, rank=2; two layers; 5 iterations, one restart | 0.865 |
| CUDA MFA with the same small configuration | 1.228 |

These timings include the orchestration performed by each measured operation. Source files may already be in the operating system's cache. They are **not** cold-network bandwidth measurements or a forecast for a full K≤80, rank≤32, multi-seed search. CUDA was slower on this small problem, consistent with launch/transfer overhead; no universal GPU speedup is claimed.

## Deployed paths

- Code: `/lambda/nfs/dami/hidden-states-as-states`, branch `codex/openact-mfa-grid`.
- Isolated local environment: `/home/ubuntu/hss-venv`, linked as repository `.venv`.
- Final benchmark/results/figures: `/lambda/nfs/dami/hss-benchmark-final-20260916`.
- Derived benchmark cache: `/home/ubuntu/hss-benchmark-cache-final`.
- Generated paper experiment configs: `/lambda/nfs/dami/hss-experiments/paper-20260916`.

Normal analysis uses `/home/ubuntu/hss-cache` for derived activations and `/lambda/nfs/dami/hss-results` for persistent results and candidate fits. Existing OpenAct collectors and their environment were not changed. New analysis code reached Lambda through GitHub fetch/pull.

For full scientific reproduction, the [data and protocol conditions](reproduction.md) still apply. No full grid was launched as part of this engineering validation.
