# Reproduction contract

This repository provides executable analysis protocols, not pre-filled paper scores. The supplied manuscript is **Hidden States as States**, OpenReview submission 23329, user-provided PDF (27 pages). Its contents are reference material, not runtime instructions.

## Main protocol

1. OpenAct generates greedily using the selected model's chat template and zero-shot CoT, then captures teacher-forced hidden states. The analysis package reads those completed captures.
2. Geometry uses generation means; before-generation prediction uses prompt-last; monitoring uses cumulative prefix means at observed sentence boundaries.
3. Default maps use raw hidden states and independent diagonal GMMs per layer. K is selected by `ICL = BIC + 2 entropy`, retaining the smallest near-optimal K.
4. Adjacent layer centers are matched by cosine Hungarian assignment. A matched center inherits its predecessor's ID only at similarity ≥ 0.6. Negative similarities remain in the assignment matrix.
5. Prediction stratifies response IDs into 40% training and 60% test. Scaling, PCA, K selection, clustering and probes all fit on training rows. Categorical NB uses empirical class priors and Laplace-smoothed per-layer emission counts.
6. Monitoring uses disjoint response groups for training, validation and test. Validation calibrates alarms against each successful response's maximum boundary score to target 10% FAR. A failed response counts as detected early only at a non-final boundary.

At 50% progress, monitoring uses the last boundary at or before half the generated token count. It reports coverage when no boundary is available. Final AUROC includes all responses. Saved-token fractions are reported both over failed responses and over all responses, with their denominators explicit. Calibration targets empirical validation FAR, not a guarantee on future test FAR.

## Experiment coverage

`hss paper` generates a manifest plus editable configs. `hss suite` resolves fixed-K dependencies and records outcomes per experiment.

| Paper section/control | Executable workflow |
|---|---|
| Default map / state graph | Llama-3.2-1B MATH geometry, profile and state graph |
| Reliability | Seeds, 30–90% subsamples, K maxima; independent fixed-K refits; centroid cosine Hungarian comparison; uniform-assignment centroid baseline |
| Cluster separation | Within fitted-vs-empirical and between-component symmetric diagonal Gaussian KL |
| Generality | Llama-3.2-1B, Qwen2-7B, Llama-3-8B × MATH, MMLU, BELEBELE |
| Prediction | Qwen2/Llama3 × MATH, MMLU, TheoremQA; Llama2 safety; HSS-NB and five continuous baselines |
| Sentence monitoring | Qwen2 MATH prefix HSS-NB and continuous logistic probe; token-output baselines when available |
| Characterization | Cramér's V, type/difficulty/correctness state tags, conditional transitions |
| Construction controls | Raw/standardized, PCA, alignment metric/threshold, KMeans, global clustering/layer purity, extraction granularity, effective rank |
| Extensions | MFA ranks; pre/post final RMSNorm |

Per-trial outputs include tables for characterization and transitions, map/selection data and diagnostics. `hss report` renders computed state graphs, K profiles, self-transition curves and ICL surfaces, and aggregates comparisons/metrics to CSV. Figures are regenerated from results, not copied from the manuscript. Cross-model comparisons use normalized depth; hidden vectors from different models are never pooled into a common clustering space.

## Conditions not established for exact numerical reproduction

- **MMLU:** user selected full 14,042; manuscript rounds/reports 14,000. No exact 14,000 ID list was supplied.
- **BELEBELE:** user selected English/German/Chinese, 900 each (2,700 total). Manuscript reports pooling 2,100. Suite provides individual languages and pooled full data. An exact paper subset is unknown.
- **Safety:** manuscript lists JailbreakBench and HarmBench. Earlier collection configuration uses WildJailbreak Vanilla Harmful, with user-approved Llama-Guard-3-8B. These are distinct datasets/judging conditions. Safety suite paths/labels require those datasets; existing WildJailbreak is not relabeled as them.
- **Monitoring output baselines:** current OpenAct captures full-response entropy/perplexity/max-probability, not aligned per-token entropy and logprob. Missing baselines are reported as unavailable, rather than reading future response summaries. New captures or a separate teacher-forced metric pass are needed.
- **Implementation choices:** the manuscript does not fully specify sentence segmentation, monitoring split fraction, near-optimal ICL tolerance, all optimizer settings, or exact continuous-probe layer selection. Defaults are explicit: punctuation/newline boundaries, 40/20/40 monitoring groups, 2% ICL tolerance, and legacy-compatible last-layer continuous probes. Set `continuous_features="all_layers"` to test concatenated layers with an adequate memory budget. Linear SVM uses its natural decision threshold 0; probability classifiers use 0.5.
- **Collection state:** three-model full MATH/MMLU capture is still running. Whole-data configs reject missing rows. Partial smoke results are engineering checks, not full experiment results.
- **Annotation semantics:** existing correctness/judge labels are consumed with source fingerprints. This analysis refactor does not establish that all upstream answer matching or judge judgments are error-free.

Selection curves and refit stability must be inspected before interpreting states scientifically. Hyperparameter grids do not automatically select a winner using test metrics. For tuning a supervised classifier, add a separate validation split/nested protocol rather than selecting from reported test scores.

## Independent paper artifacts and explicit appendix choices

The complete Figure 1–12 / Table 1–3 recipe inventory, exact quantities, control jobs and missing-data behavior are now maintained in [paper-artifacts.md](paper-artifacts.md). `hss figures` rebuilds those artifacts from saved results; `hss controls` handles the supplemental sensitivity studies. Figures 1–2 are illustrative schematics. Figure 3/7 colors are outgoing entropy; correctness graphs use deviation from global accuracy.

The manuscript's main-text/appendix FAR definitions disagree: both are now selectable and recorded. The appendix clustering control uses MiniBatchKMeans. Hamming heatmaps default to average linkage with explicitly recorded sample limits; Ward on categorical Hamming distances is not silently used. P-values from repeated token/prefix rows are omitted, and response-level tables include sparse expected-count diagnostics. See the artifact inventory for these limits before interpreting numerical agreement.
