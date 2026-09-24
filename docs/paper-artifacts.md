# Paper figures, tables and robustness controls

Reference: user-supplied **Hidden States as States**, OpenReview submission 23329 (27 pages). Recipes implement the quantities described in that manuscript; they do not embed its reported numbers or claim pixel-identical figures.

## Figure and table inventory

| Recipe | Required suite jobs | Computation |
|---|---|---|
| `figure_01` | None | Illustrative state-abstraction schematic |
| `figure_02` | None | Illustrative data/fit/alignment/evaluation pipeline |
| `figure_03` | `default_map` | Llama-3.2-1B MATH generation mean; nodes scale with counts, color is outgoing entropy; last layer neutral |
| `figure_04` | `default_map` | Sorted marginal state frequency + cumulative mass; Hamming trajectory similarity; active K/self-transition/uniform 1/K |
| `figure_05` | `qwen_math_map`, reliability jobs and nine cross-model/dataset maps | K trends, fixed-K refit center distances, random-assignment baseline, symmetric KL, normalized-depth profiles |
| `figure_06` | `qwen_math_map` | Cramér's V across type, difficulty and correctness; chi-square and sparse-cell diagnostics exported |
| `figure_07` | `kmeans_control` | MiniBatchKMeans state graph, entropy colors |
| `figure_08` | `kmeans_control` | Same map, state accuracy minus global accuracy |
| `figure_09` | `qwen_math_map` | Relative ICL heatmap: x=K, y=layer, selected/near-optimal overlay |
| `figure_10` | `qwen_prompt_map` | Full-data prompt-last geometric map with correctness coloring |
| `figure_11` | `qwen_math_map` | Full-data generation-mean map with correctness coloring |
| `figure_12` | `qwen_math_map` | x=global state ID, y=layer, correctness deviation, first/last appearance markers |
| `table_01` | Eight `prediction_*` jobs | Long and pivoted AUROC/accuracy tables for saved held-out scores |
| `table_02` | `sentence_monitoring` | Prefix/final AUROC, response FAR, early detection and token-saving metrics; missing baselines explicit |
| `table_03` | Available results | Actual model/dataset/representation inventory and resolved protocol parameters |

The graphs center each layer's nodes vertically while retaining expansion/contraction in the number of states. Node labels appear up to 300 nodes; exact node/edge tables are always saved. Color ranges, units and entropy at the final layer are explicit. State tags are descriptive annotations, not labels used for clustering.

### Important choices and inconsistencies

- **Trajectory heatmap:** Hamming dissimilarity; average-linkage default. The manuscript cites Ward; Ward requires Euclidean distances, so we do not silently apply it to categorical Hamming distances. CLI supports average/complete/single. O(N²) computation uses up to 1,000 deterministic trajectories by default; `--max-trajectories 5000` selects the full default MATH map. Exported row IDs expose the subset.
- **ICL normalization:** `(ICL - min(ICL)) / max(abs(min(ICL)), 1)` within each layer; smallest K within configured tolerance (default 2%) wins. The manuscript does not fully specify all numerical optimizer/tolerance details.
- **Center reliability:** every pair of fixed-K refits, Hungarian assignment with cosine distance in original feature coordinates. Unmatched counts and ARI on shared rows are exported. Random baseline means *uniformly assign samples, recompute centroids, then match*; random pairing of existing centroids is a different diagnostic.
- **Variation bands:** K profiles use mean ±2 sample SD; center distances use mean ±1 sample SD across refit pairs. These describe procedural variation and are not confidence intervals. Pairs sharing a refit are dependent.
- **Subsampling:** text says 30–90%, while the figure legend differs. Generated settings explicitly use 0.3/0.5/0.7/0.9, seeds 42–46 and Kmax 20/40/60/80.
- **FAR:** main text includes all boundaries; Appendix C.2 excludes the final one. `evaluation.far_scope="all_boundaries"` is the default; `"nonfinal"` is a separate control. Both calibrate at response level, including successful responses with no eligible nonfinal boundary in the denominator. Early failure detection always excludes the final boundary.
- **Inference:** chi-square p-values are omitted for repeated prefix/token/sentence rows because these are not independent responses. Sparse expected counts are reported even for response-level tables; a p-value is not an automatic significance claim across many layers/tests.
- **Bootstrap:** optional stratified bootstrap of held-out *responses*, with fixed trained model. It measures test-sample uncertainty, not training/refit variance. Seed controls separately measure procedural sensitivity.

## Reproduce or rebuild selectively

```bash
# Inspect availability without fitting or reading activation tensors.
hss data configs/lambda/datasets.toml

# Pin dataset paths once, generate explicit protocols into a persistent study.
hss paper --catalog configs/lambda/datasets.toml \
  --directory /lambda/nfs/dami/hss/studies/paper \
  --cache-root /home/ubuntu/hss-cache \
  --artifact-cache-root /lambda/nfs/dami/hss/cache/fits \
  --output-root /lambda/nfs/dami/hss/results/trials --check-data

# Run the requested map or study jobs; same command resumes after interruption.
hss suite /lambda/nfs/dami/hss/studies/paper/suite.json --only default_map

# Audit, then build only figures whose dependencies have completed.
hss results /lambda/nfs/dami/hss/results/trials --validate --full
hss figures /lambda/nfs/dami/hss/results/trials \
  --suite /lambda/nfs/dami/hss/studies/paper/suite.json \
  --destination /lambda/nfs/dami/hss/figures/paper \
  --only figure_03 figure_04 --max-trajectories 5000

# Check complete-paper coverage without plotting or fitting.
hss figures /lambda/nfs/dami/hss/results/trials \
  --suite /lambda/nfs/dami/hss/studies/paper/suite.json \
  --destination /lambda/nfs/dami/hss/reports/coverage --check --strict
```

The generated `index.html` is a local gallery; each figure has a separate manifest and CSV/NPY inputs. Open it from a mounted/downloaded result directory. Rebuilding plots does not rerun clustering.

## Supplemental controls

`hss controls ROOT --suite SUITE --destination DEST` emits per-control profiles, numerical diagnostics and comparisons. `--only JOB ...` selects controls. `--bootstrap 1000` adds fixed-model response bootstrap intervals for prediction results.

| Study jobs | Checks |
|---|---|
| `reliability_trends`, `reliability_centers` | Seeds, response subsamples, K-range changes, equal-K matched center distances and ARI |
| `preprocessing_control` | Raw vs standardized, each across seeds 42–46 |
| `pca_control` | No PCA vs 64/128/256/512; K, self-transition, center distance, ARI |
| `alignment_control` | Cosine / 1/(1+Euclidean distance), eta 0.3–0.7; inheritance and global state counts |
| `kmeans_control` | MiniBatchKMeans + sampled silhouette |
| `global_clustering_control` | Stack all layers, fit one clustering, report component layer purity |
| `granularity_control`, `token_granularity_control` | Mean/prompt/sentence/prefix; token geometry is a separate expensive job |
| `*_prediction_control` | Preprocessing, PCA, alignment, cluster method and seed sensitivity of held-out HSS-NB AUROC/accuracy |
| `monitoring_far_scope_control` | Main-text and appendix FAR definitions |
| `rms_control`, `mfa_control` | Pre/post final RMS and MFA rank extensions |

Each control exports exact settings, metrics, global-state count, inherited fraction, response IDs and comparisons. Same-representation/snapshot center comparisons avoid mixing incompatible spaces; RMS and different-granularity profiles remain separate when snapshots differ. Tuning on the final test scores would bias the reported evaluation; these grids are sensitivity studies, not automatic model selection.

## What still requires data or specification

A successfully rendered smoke figure establishes the software path only. Full MATH/MMLU collection is ongoing. TheoremQA and BELEBELE collection are not launched by HSS. User choices (MMLU 14,042; EN/DE/ZH BELEBELE 2,700) differ from paper counts (14,000; 2,100). Exact subset IDs were not supplied. Paper safety requires JailbreakBench/HarmBench rather than current WildJailbreak. Prefix entropy/logprob baselines require aligned per-token metrics absent from current captures. These are recorded in coverage/protocol notes and are never silently substituted.
