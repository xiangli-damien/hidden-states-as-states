# Revision experiment execution state

## 2026-09-23 07:01–07:30 UTC — audited first results and strengthened intervention design

Read [first result interpretation](revision-first-results-20260923.zh-CN.md). Original geometry finished all 64 views / 768 converged fits in about 36 minutes; every selected ICL and decoder/summary SHA was independently verified. Sixteen views choose the grid boundary K=64; no global-optimum claim. Paired historical-test results: block28/prefix16 linear-probe mean16 0.7979 vs last 0.7099, delta 0.0880 [0.0633,0.1118]; prompt mean advantage CI crosses zero. Discrete mean16 state adds only 0.0043 [−0.0046,0.0138] above nuisance at that block/prefix. Pointwise, exploratory, not causal. Files/figure copied to local `results/revision-foundations-20260923`.

**Important intervention amendment before any patch outcomes:** original token codebooks sampled fixed positions 0/5/10/15. All-16 reconstruction could therefore encounter systematically underrepresented template positions. Original fits are retained as a sampling comparison. New root `/lambda/nfs/dami/hss/revision-token-decoders-v2-20260923` fits every training position at the six required token views (prefix0 chat/question, prefix16 generated; blocks14/28), K grid and seeds unchanged, three CPU workers × two threads. Initial PID428820; check `fit_status.json`. This also implements true empirical-mean-centered local PCA, separate from fixed-GMM-center residual SVD. Updated foundation config points forthcoming GPU interventions to this v2 root and enforces 16 positions per question, SHA checks and exact condition coverage. No old patch result is overwritten; none had run yet.

GSM8K full collection continues normally, ~192 published at the first check and increasing, GPU ~15GiB/40GiB, SSD ~126GiB free. The primary supervisor still schedules functional then behavior after GSM8K, with geometry already complete. Its geometry child may remain marked running until the supervisor reaps it; trust the geometry's own complete marker/log plus exit status, not that stale parent display alone.

**Transfer implementation ready:** `revision_transfer.py prepare/evaluate` and `run_revision_transfer_queue.py`. The follow-up queue waits for the primary GPU queue to terminate, checks complete GSM8K, prepares target rows, extracts matching prefixes/real norm confidence, then fits/evaluates 36 preregistered view/block/prefix combinations on CPU. Roots `/lambda/nfs/dami/hss/revision-gsm8k-20260923` and its `queue_status.json`. Five regimes: source-full frozen; source-full map with target-adaptation readout; target refit; source map/readout fit to a hash-chosen source subset matched to target adaptation count; matched-source map with target readout. Raw diagonal GMM, train-only ICL, no target confirmation fitting. Target adaptation mean is the common geometry denominator. Coverage thresholds use the relevant domain's validation inputs. Source continuous LR and current entropy are baselines; four-block NB is secondary and counts prior once.

Primary new-dataset hypothesis is frozen **block28/prefix16 mean16 versus last** NB AUROC on identical GSM8K confirmation questions, selected from source-only findings before target state analysis. All 36 views reported; target confirmation is not searched for the best layer/K/representation. Two operational smoke questions were in confirmation: report method-held-out status accurately, not never-inspected labels. Exact source–target prompt overlap is checked before adaptation.

Deployment remains local tests → commit/push → Lambda fetch/ff-only. Semantic amendments use separate namespaces; original geometry remains immutable. Latest first-night queue documentation above supersedes the original “4-position interventions / PCA pending” descriptions below. Controlled donor/steering and token occupancy/order/input-text controls remain outstanding; do not mark them complete.

## 2026-09-23 06:26 UTC — first real queue running

Protocol: [Chinese full checklist and premise tree](revision-execution-plan-20260923.zh-CN.md).

- Local HSS `/Users/lixiang/Projects/hidden-states-as-states`, branch `codex/openact-mfa-grid`.
- Remote HSS `/lambda/nfs/dami/hidden-states-as-states`; experiment `/lambda/nfs/dami/hss/revision-foundations-20260923`.
- OpenAct `main`, collection code commit `44b029d`; HSS queue code `3ecb219` (later report/document-only changes possible).
- All deployment used local commit → GitHub push → Lambda fetch/ff-only.

## Completed prerequisites, not scientific findings

1. MATH 5,000 fresh unpadded prefix captures finished. Blocks 7/14/21/28, prefix 0/16/64, actual 16-token tail windows, separate question-tail windows and means. Pre-final-norm block outputs. Short generated sequences are explicitly unavailable at later prefixes.
2. GPU identity audit passed at four blocks × widths 1/4/16: exact next-token logits; eight-token identity continuation exact. Additional tiny-Qwen tests check complete reference NLL, cache behavior, explicit positions and matched perturbation energy. Seven unique foundation tests passed across CPU and GPU environments; each environment skips unavailable dependencies.
3. Current-prefix entropy/margin/max probability/RMS computed with the actual pinned final norm and unembedding. Batched bf16 readout rounding is recorded; this is not a fitted confidence-direction surrogate.
4. GSM8K main/test prepared, exact pinned revision, matched MATH prompt-template hash `507f30ec8346341e`; 1,319 rows, hash split 541 adaptation / 242 validation / 536 confirmation. Original benchmark train split is not used in the transfer experiment.
5. Two-question GSM8K collection smoke passed actual generation, all-layer activations, pre/post norm, labels, exact fixed-shape causal replay and SHA-verified publication. These two items form a separate smoke run, not additional independent test examples.

## Running / queued

- Queue PID initially 427701; read live `queue_status.json`, never assume old PID still valid.
- CPU GMM geometry fitting started (PID initially 427844), four workers × two BLAS threads, 64 view/block/prefix combinations, K 1/4/8/16/32/64 × two seeds, convergence checked. Boundary-optimal K is flagged. Fit calculations use float64 raw values; no input normalization.
- Full GSM8K collection started (PID initially 427935), `/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923`, SSD `/home/ubuntu/openact-gsm8k-20260923`, 32-row shards, 50 GiB reserve. Smoke uses a separate `gsm8k_transfer_smoke_20260923` tree.
- Next GPU stages: same-prefix full-reference NLL/KL reconstruction intervention pilot; separate free-generation task-accuracy pilot. Each condition uses fresh prefill/KV; real-token codebooks only. Functional stage selects 32 validation + 32 historical-test questions; behavior selects 12 + 12, fixed by question hash. These are pilots, not new confirmatory MATH tests.
- Heartbeat `hss-token`, 30-minute interval, only notifies on meaningful changes / audited stage outcomes / failures; continues the remaining authorized high-priority experiments after prerequisites.

## Definitions and remaining implementation gaps

- Source `label=1` means **correct**; predictive failure target is `1-label`. Split names are exactly `train/validation/test` (3,011/1,003/986).
- Actual-token codebooks fit four fixed positions per training question (12,044 vectors); reconstruct all 16 held-out positions. Mean-vector codebooks are distinct. No claim of fitting every MATH token.
- Key `local_pca_r` currently implements rank-r SVD of nearest-assigned train residuals **about the fixed GMM component center**. Reports call this `local_residual_svd_r`. It is not empirical-mean-centered local PCA and not MFA. Add a true per-partition empirical-mean PCA baseline in a separate, explicitly versioned follow-up before claiming that exact comparison is complete. Global PCA is centered at the training mean.
- Current representation classification compares last / mean4 / mean16 / prefix-wide mean with a state NB, regularized full-vector LR, and current-information nuisance controls. Token occupancy / ordered readouts and input-text semantic baselines still need implementation. Geometric token reconstruction is already implemented.
- Cross-prefix comparisons need the intersection of available question IDs; later prefixes exclude shorter responses. Report script includes that prediction table. Full-answer means must remain separately labeled post-hoc.
- Selective steering, controlled donor counterfactuals, frozen-map GSM8K evaluation and online FAR are **not in the initial executable queue**. The checklist/heartbeat directs their subsequent implementation and gated execution; none is marked complete.
- Raw GMM-selected codebooks have both nearest and posterior assignment audits. Their agreement is not assumed. Geometry quality and classification do not establish a functional or causal state.
- Queue currently waits for all geometry views after GSM8K. If that leaves the GPU idle, run required-decoder readiness scheduling, preserving active CPU jobs and avoiding duplicate GPU stages. Do not kill a healthy child simply to alter the supervisor.
- Strict stage provenance includes code/config/input hashes and git commit. If restarting after another commit, inspect existing plan first; preserve output identities and do not overwrite an incompatible scientific stage.

## Reporting

`scripts/report_revision_foundations.py --config configs/revision_foundations_20260923.json` generates actual partial or complete tables. No synthetic values enter reports. It includes question-level pointwise intervals, paired accuracy changes, failure-to-success / success-to-failure counts, answer agreement, parse/truncation, and raw losses for loss-recovered denominators.

Copy only lightweight reports/metadata back to `results/revision-foundations-20260923`; raw tensors stay on dami. Add a dated entry after each audited result with effect size, controls, scope and failure cases. Existing MATH historical-test results remain exploratory. GSM8K's confirmation split is the first new frozen evaluation opportunity.
