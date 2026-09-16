# Configuration

TOML and JSON use the same dataclass schema in `src/hss/experiments/config.py`. Unknown fields and unknown dotted overrides fail. `extends = "base.toml"` resolves relative to the child file; data/output paths resolve from the working directory, with `~` and environment variables expanded. CLI values are JSON: quote strings, use `true`, and use `null` to clear optional values.

| Group | Parameters and choices |
|---|---|
| `data` | `paths`, `expected_model`, `expected_samples`, `representation=mean/prompt_last/prefix/tokens`, `final_norm=pre/post`, `layers`, `label`, `label_file`, `max_samples`, `max_shards`, `only_valid`, `exclude_truncated`, `require_copy_receipt`, `batch_tokens`, `min_free_gib`, optional token metric paths |
| `cluster` | `method=gmm/mfa/kmeans`, fixed `k` or `k_values` or `k_min/k_max`, `rank`, `covariance_type`, `reg_covar`, `n_init`, `max_iter`, `tol`, `chunk_size`, `backend`, `device`, adaptive regularization options, initialization, parsimony tolerance, `assignment=nearest/posterior` |
| `transform` | `standardize`, `pca_components`, `whiten`; fitted on map-training rows only |
| `alignment` | `similarity=cosine/euclidean`, `method=hungarian/greedy`, inheritance `threshold` |
| `evaluation` | mode, split seed/fractions, Laplace alpha, FAR, positive-label polarity, probe list/features/scaling/C/MLP sizes/iterations, fit fraction, fixed map or fixed K reference, diagnostics |
| `execution` | raw-data cache, fit-cache and output roots; workers, BLAS threads, RAM budget/reserve, GPU reserve, task index/count, retry policy, max trials |

Negative layer indices select positions from the captured layer list; nonnegative values are actual layer IDs. OpenAct layer 0 is the embedding output. A typical Llama-3.2 run has 17 positions: embedding plus 16 decoder blocks. `final_norm=pre` changes only the final decoder output, using OpenAct's explicit `final_norm/pre` arrays. Intermediate layers retain their captured block-output meaning.

`label_file="correctness"`, `label="is_correct"` joins `labels/correctness.parquet` on `sample_idx`. Unknown labels remain unknown; supervised modes require complete binary labels. Safety can select `label_file="safety"`, `label="is_safe"`; set `positive_label=0` only if reversing the input binary label is intentional.

`token_entropy_path` and `token_logprob_path` are **Zarr keys**, not filesystem paths. Each must hold one value per generated token in the same order as `tokens/sample_ptr`. Prefix baselines sum only values already observed. Whole-response entropy/perplexity cannot serve as prefix measurements.

## Grids

```toml
[grid]
"cluster.rank" = [4, 8, 16]
"seed" = [42, 43, 44]
"alignment.threshold" = [0.4, 0.6, 0.7]
```

This creates 27 trials. K-selection scans are inside each trial. Explicit fixed `cluster.k` overrides K search. `cluster.k_values` overrides the min/max range. The smallest candidate within `parsimony_tolerance * max(abs(best_ICL), 1)` is selected. Set tolerance to zero for the exact minimum. KMeans uses negative sampled silhouette, not ICL.

One-factor perturbations can use JSON `grid.variants`, a list of objects with dotted keys, instead of a Cartesian product. Duplicate resolved configs are removed. Execution limits are properties of the invocation and cannot be grid axes.

```bash
# Use a coarse, explicitly changed search during exploratory work.
hss sweep configs/mfa_grid.toml \
  --set 'cluster.k_values=[2,4,8,16,32,64,80]' \
  --set 'execution.workers=4' --set 'execution.threads_per_worker=2'
```

This changes the searched model family and is an exploration, not an exact full K scan.

## Checkpoints and provenance

Data snapshots fingerprint manifest, labels, row metadata, success/copy receipt, reader code and data selection. Published activation chunks are treated as immutable under OpenAct's verified publication contract. Reader preparation does not re-hash terabytes of activation chunks on every launch.

Algorithm source and installed numerical-library versions enter experiment/cache identity. Split rows, resolved config, source snapshot, candidates, fitted parameters, projections, alignments, states, predictions and metrics are saved. Model arrays use NPZ with `allow_pickle=False`; data matrices use contiguous per-layer NPY memmaps. Full outputs can be analyzed without loading language-model weights.

A saved sweep plan freezes its source snapshot. Rerunning it resumes that snapshot even as collection publishes new shards. `--refresh-data` explicitly rebuilds the plan against available data. A failed or interrupted trial never writes `_SUCCESS.json`; completed per-K fits remain reusable. `retry_failed=false` records skipped failures and returns a failing status instead of declaring success.

`fixed_k_map` refers to a prior trial's `selection.json`, permitting independent refits with the same K. `fixed_map` uses an existing compatible map for geometry only; supervised protocols always fit their own training map. External reference file contents enter trial identity.
