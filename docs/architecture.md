# HSS architecture and result lifecycle

HSS is the analysis package. OpenAct owns generation, activation capture and upstream task labels. HSS never loads a language model to cluster saved activations. The two repositories and virtual environments remain separate on Lambda.

## Module boundaries

```text
OpenAct published Zarr + Parquet       Other collector: NPY + Parquet
                │                                  │
                └──────────── hss.data ─────────────┘
                              │ CachedStates (local mmap)
                    hss.transform + hss.cluster
                              │ portable fitted arrays
                       hss.experiments
                    split → fit → align → evaluate
                              │ immutable trial directory
                         hss.results
                     audit / catalog / load model
                              │
                         hss.analysis
                    derived tables / comparisons
                              │
                           hss.viz
                  PNG + SVG + CSV + render manifest
```

| Package | Responsibility | Inputs → outputs |
|---|---|---|
| `hss.data` | Source adapters, immutable selection, local cache | `DataSpec` → `CachedStates` |
| `hss.transform` | Fit scaling/PCA and invert centroids to raw coordinates | Fit rows → portable `Projection` |
| `hss.cluster` | GMM, MFA, KMeans, MiniBatchKMeans | Matrix + parameters → `ClusterModel` |
| `hss.experiments` | Protocols, fitting checkpoints, resource limits, grids | Resolved config → completed trial |
| `hss.results` | Portable reading, discovery and checksum/structure audit | Trial directories → `Result` / `ResultCatalog` |
| `hss.analysis` | Hamming trajectories, entropy, profiles, refit comparisons, bootstrap | Saved trials → numerical tables |
| `hss.viz` | Pure plotting functions, paper recipes and control reports | Tables → figures + provenance |

`experiments.openact` remains a compatibility import; its implementation lives in `hss.data.openact`. Projection and model loading no longer depend on the experiment runner. Legacy root `experiments/`, `hss_workbench/` and `hss_bundle_bridge/` support existing notebooks; they are not the reproduction entry points.

## Data contract

`CachedStates` contains float32 NPY matrices of shape `[rows, hidden_dim]`, one file per selected layer. `rows.parquet` records `sample_id`, response `group_id`, label, `token_end`, `n_tokens`, `is_final`, source and available type/difficulty/language metadata. Matrices and metadata always share row order. For subset runs, profile `relative_depth` normalizes the ordinal position in the selected-layer sequence; use the explicit layer IDs for absolute decoder depth. Main-paper cross-model jobs select all available layers. Models/revisions, layers and dimensions must agree across shards; duplicate responses, invalid labels, missing full-dataset coverage and nonfinite activations fail preparation.

Representations are distinct:

- `mean`: generation-token mean, one row per response.
- `prompt_last`: final prompt token, one row per response.
- `prefix`: cumulative mean through each observed sentence boundary.
- `sentence`: mean of the current sentence span only, excluding earlier sentences.
- `tokens`: one row per generated token.

`final_norm="pre"` replaces only the **final decoder layer** with its pre-final-RMS capture. It does not mean pre-RMS at every block. Prefix and sentence computations accumulate in float64 and export float32. The final token span is retained even without terminal punctuation.

Other collectors use the interchange writer:

```python
from hss.data.arrays import export_arrays
from hss.data import DataSpec, prepare

# Each matrix has identical (N, D) shape. rows is a pandas DataFrame.
export_arrays("/path/new_dataset", {0: layer0, 16: layer16}, rows,
              model=["model/id", "pinned_revision"], representation="mean",
              final_norm="post", dataset_id="my_dataset")
data = prepare(DataSpec(["/path/new_dataset"], source_format="arrays",
                        dataset_id="my_dataset", min_free_gib=16), "/fast/cache")
```

For prefixes/tokens/sentences, supply ordered integer `token_end` and `n_tokens`, including the final boundary. All boundaries for a response must have the same label and token count. `label` or the configured label column must be binary; unknown labels are allowed for geometry only. Arbitrary collector arrays cannot prove causality: their producer must supply the declared representation correctly. Full-response uncertainty is never substituted for prefix uncertainty.

## Trial contract: save once, inspect independently

```text
results/trials/<trial_id>/
  config.json                 Resolved scientific + execution parameters
  data_snapshot.json          Selected source files, hashes, model/revision, rows
  states.npy                  Global discrete state IDs [N, layers]
  rows.parquet                Row identity, labels, boundaries and metadata
  split.npz                   train / validation / test / map_fit row indices
  alignment.json              Per-layer local-to-global vocabularies
  selection.json              All candidate criteria and selected K
  models/layer_*/              Model + projection NPZ and JSON (no pickle)
  diagnostics.json            Dispersion, separation and optional effective rank
  associations.csv            Cramér's V, chi-square, p-values and sparse-cell diagnostics
  state_tags.csv              Type and correctness tags
  transitions.csv             Overall and label-conditioned transitions
  predictions.parquet         Held-out scores, when supervised
  evaluation.json             Metrics + unavailable baselines, when supervised
  monitor_*.parquet           Response-level alarm summaries, when monitoring
  summary.json / status.json  Identity, versions, timings and outcome
  artifacts.json              SHA-256 and size inventory of completed artifacts
  _SUCCESS.json               Written last, after the artifact inventory
```

Models and projections reload without the original activations, fit cache or Python pickle. A moved result directory remains readable; historical absolute paths in the snapshot are provenance, not required file locations for rendering. Continuous probe scores are saved for evaluation/replotting; **continuous sklearn estimators are not currently serialized for future inference**. Frozen geometric maps can be applied to compatible new data with `evaluation.fixed_map`.

```python
from hss.results import Result, ResultCatalog, load_layer
from hss.analysis.tables import state_map, trajectory_similarity
from hss.cluster.assignment import assign
from hss.viz.plots import state_graph

r = Result("/persistent/trials/TRIAL_ID")
assert r.validate(full=True)["valid"]
model, projection, selection = load_layer(r.path, r.layers[-1])
new_states = assign(model, projection.transform(new_vectors),
                    r.config["cluster"]["assignment"])
nodes, edges = state_map(r)
fig = state_graph(nodes, edges, color="entropy")
fig.savefig("/separate/figure.svg")
```

`assign()` returns local component IDs and preserves the recorded nearest/posterior policy. Convert with that layer's `alignment.json` vocabulary if global IDs are needed. For complete new-data evaluation, use `hss run` with the frozen map so the configured assignment policy is preserved.

## Cache boundaries

Data, numerical fitting, completed trials and rendering have separate source identities. Figure/style edits leave data, fit and trial identities unchanged. A changed optimizer invalidates affected fit identities; changed evaluation code creates new trials while reusing identical fits. Reader changes create new frozen plans/data caches. Inputs, fitting rows, projection parameters, K, model method/backend and seed participate in fit identities.

Changing eta, an evaluation/report option or near-optimal selection tolerance reuses fitting checkpoints. Changing seed/rank/PCA changes the fitted model. Changing a dataset selection creates a different immutable snapshot. Legacy results remain readable; this module reorganization changes old code fingerprints once and does not rename old experiments as new ones.

`hss results ROOT --validate` checks structure, coverage, vocabulary and response split isolation plus inventoried file sizes. `--full` additionally hashes every recorded file. Old results without inventories are explicitly marked legacy, not asserted checksum-verified. Warm sweep resume audits completed artifacts before returning cache hits.

## Decoupled rendering

`hss figures`, `hss controls` and `hss report` read completed result files only. A render directory contains PNG/SVG files, the exact numerical CSV/NPY inputs, chosen sample IDs/order, plotting parameters, input trial IDs and inventory hashes, output hashes and figure code version. Rendering into an immutable trial directory is rejected.

The paper recipe resolves suite job aliases through `suite_status.json`. Two identical jobs may share a trial ID even when their human-readable names differ; no figure depends on whichever alias happened to finish first. `coverage.json` distinguishes rendered, missing inputs, invalid inputs, partial baselines and failed rendering. `--strict` makes incomplete paper coverage a nonzero exit.

## Extension points

1. New collector: implement a `DataSpec` adapter returning the same cache contract, or export the array interchange format.
2. New cluster method: implement `ClusterModel`, portable model arrays/rebuild in `cluster.registry`, fitting dispatch in `cluster.methods`, and the selection criterion/storage estimate. Test likelihood/predictions after reload.
3. New experiment: keep fitting/split logic in the runner and emit a self-contained artifact. Register its config in the paper/study suite.
4. New statistic: add a function to `analysis/` reading `Result`, independent of plotting and fitting.
5. New figure: write a pure function in `viz/plots.py`, then register its inputs and exports in a recipe. It should be testable from saved results after deleting raw data.
