# HSS sink follow-up v1.1 — executable review

This document amends, rather than replaces, `HSS_followup_spec_v1.original.md`.
The originals are preserved verbatim. The only authoritative numerical parameters
are `scripts/hss_followup_tools.py::PARAMS`; source, parameters, input files and UTC
freeze time are recorded in the run's `plan.json` and `FREEZE.json` before Step 1.

## What is being executed

1. After the predecessor completes, audit its 529 records. Unhooked re-encoding
   of its 465 test answers and 155 historical answers: full response means and
   sentence-prefix means at all 29 stored indices. Short forced-answer suffixes
   are the sole new generation in Step 1 and are **unhooked in every arm**.
2. A/B question split of the predecessor's 3,011 training questions. Build new
   directions and thresholds on A; all 35 likelihood conditions on B. Retain
   the full per-question table, geometry diagnostics and every control.
3. Run the supplied D6 decision without manual override. A selected C1/C2/C3
   enters the full 155-question 3A comparison if it meets the time gate.
   If the decision selects 3B, execution requires the **original Qwen prefix
   map, HSS-NB parameters and threshold**. These are currently unlocated.
   Existing Llama monitors cannot substitute for them. No new alarm is trained
   and represented as the original. Missing artifacts are an explicit blocked
   stage, not a completed retry experiment.

## Verified bindings and necessary clarifications

- Qwen2-7B-Instruct, pinned revision from predecessor, HF BF16 SDPA, batch 1,
  greedy, 2,048 tokens. Block 14 = `model.model.layers[13]` output = stored
  `hidden_states[14]`. Preflight compares against actual OpenAct tokens and
  activations (relative error < 1e-3), identity hook, causal log-probability
  alignment, generated-position mask, and numpy/torch equivalence (< 1e-4).
- Response GMM: existing raw post-view means, diagonal covariance, 2% ICL
  selection per layer, MAP assignment. Re-export the coherent alignment using
  the existing cosine/Hungarian implementation at eta=.6. No GMM refitting.
  Its block-14 local sink 8 has global ID 24, present deepest at index 27.
  The legacy paper's sink ID has not been verified; IDs from other K schedules
  are not interchangeable. Index 28 is post-final-RMSNorm; 27 is a block output.
- Token chart: existing `p16_l14_tokens` decoder, 64 centers, nearest Euclidean
  assignment, first eight columns of its local basis. No token normalization.
- The reused response map, control centroids and token chart were fitted before
  the new A/B split. They are **exceptions** to an absolute A-only fit claim.
  New d, type-target vectors, tau and control taus use A only. We preserve the
  specification's frozen-centroid manifold controls and record this exception,
  rather than quietly replacing the centers with another estimator.
- C2 targets cover every dataset type (including types in normal controls),
  not merely types represented in sink B. Correctness-guided C2 is supervised.
- Cohort IDs are sorted before deterministic stratified splitting. Explicit
  source question IDs, control pairs, vectors, thresholds and seeds are saved.
- Sentence boundaries exactly reproduce current repository function
  `hss.data.openact.sentence_boundaries`. That function explicitly documents
  that the manuscript's original sentence tokenizer was unspecified. This is
  **not** a verified match to the original paper's segmenter. Tests compare
  the implementation to the current source. Saved token IDs are never replaced
  by a text roundtrip; where BPE roundtrip differs, prefix decoding supplies
  monotone character offsets with explicit checks and a logged rule.
- Means and original-answer NLL include saved terminal EOS, matching the source
  mean map. EOS is removed only when appending FORCE. Cut positions exclude
  terminal special tokens. Gold answer tokens are scored using preceding
  positions. Normal-answer score spans boxed content through its closing brace.
- Keep the supplied FORCE string; report training boxed-format counts before
  execution. This is a fixed diagnostic prompt, not selected on B performance.
- Projections and log probabilities are F32. Stored model activations and model
  updates remain BF16; report actual nonzero updates separately from ideal
  clamp activation, since tiny ideal changes can round to zero.
- Prefix arrays are per-answer NPZ shards, indexed and checksummed by
  `step1_prefix_states.manifest.json`, rather than one multi-GB mutable archive.

## Decision and inference corrections

- A tie requires **both** CI lower <= 0 and CI upper >= 0.
- Mandatory metrics and the entire 35-condition question grid must be complete
  and finite. Invalid required records stop execution. Only the prespecified
  short-loop metric may be missing. No silent selection of complete-case
  subsets with different questions for different candidates.
- Empty normalized votes are invalid. Degenerate/nonfinite directions fail.
- Keep P3 exactly as supplied: no worse than the worst matched control and
  >= -0.10 nats mean **summed answer log probability**. This is neither a
  per-token threshold nor a safety noninferiority guarantee.
- Keep the supplied C0 gate as a conservative consistency rule. C0 uses A-only
  d, whereas the predecessor used full-training d. Failure of that gate is not
  a controlled proof that likelihood screening is uncalibrated.
- Positive publication gate: candidate must **outperform** its control AND
  two-sided exact McNemar p < .05. A significant worsening is not success.
- The supplied 1/11 calculation needs an exchangeability assumption which
  isotropic, centroid-difference and learned vectors do not guarantee. Do not
  present it as measured type-I error or familywise error control.
- Preserve the 2026-09-26 01:59 UTC freeze and one-hour reserve. Estimate 3A
  using observed predecessor seconds/token, all 155 cases, two new arms plus
  possible full zero replay, max token budget and 25% runtime headroom. No
  cohort shrinkage to make the time gate pass. Post-freeze results are marked
  outside the submission edition.

## Execution and outputs

CPU preparation: `scripts/prepare_hss_followup.py`.
GPU preflight/Steps 1–2/conditional 3A: `scripts/run_hss_followup.py`.
CPU complete-cohort audit, D6 decision and reports: `scripts/report_hss_followup.py`.
The worker retains fresh caches, fixed order, per-record receipts, resumable
outputs, progress counters and failures. It never launches a second model while
another GPU process is active. No dependency upgrades are required.

The original experiment was inspected during development, and this entire
follow-up is exploratory. All positive and negative results are retained.
