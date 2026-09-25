# Review log — 2026-09-25, before follow-up model execution

The exact UTC source freeze is saved by preparation in `plan.json`/`FREEZE.json`.
This log accompanies the original attachments and v1.1 reviewed specification.

- Fixed the helper's incomplete interval-containment check for candidate ties.
- Reject incomplete/nonfinite mandatory grids and degenerate direction vectors;
  normalized-empty votes cannot win. Move SciPy import to CPU statistics only,
  so the pinned GPU environment needs no installation or upgrade.
- Made explicit previously implicit layer, rank, deadline, control fraction,
  sample count and decoding parameters in PARAMS, without changing their values.
- Documented frozen-map / chart / control-centroid reuse as the exception to
  strict A-only representation fitting; new estimands remain A-derived.
- Resolved the current map's sink identity with its actual 2% K schedule:
  local 8 at block14 -> global24, deepest matching layer27. It is not asserted
  to be the unlocated original paper's sink.
- Disclosed current repository sentence segmentation as a documented surrogate;
  cannot honestly assert the original paper used that exact function.
- Added explicit EOS continuation rules and exact saved-token alignment checks.
- Prefix storage is sharded with a manifest for resumability.
- Required positive direction of effect for the significant primary paper gate.
- Stage3B remains conditional on locating the original Qwen alarm artifacts;
  no Llama model or newly trained alarm is silently substituted.

The supplied synthetic self-tests are implementation checks, **not empirical
experiment results**. GPU preflight and actual screen outcomes are reported
separately after they exist.
