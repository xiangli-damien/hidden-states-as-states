# Online generation numerical amendment v2

Original exp4 stopped at 50/606 records on math_4961 / ORTH_MAN_1, at one token with C1 target norm 2.69552238e-5. The token absolute norm error was 1.0349e-7 (inside the original token bound), but relative step energy error was 0.007664, above the frozen 0.002 gate. All original files and failure.json remain unchanged.

The reproducible failed activation is recorded in exp4/rounding_diagnostic; SHA256 a8a67e30769ffec5190cb72dd41192a8e2e35391e1a30ddf6bca8e471c65ed7a. Removing only the internal 1e-7 numerical stopping floor reduced the norm error to 2.1701e-9 and energy error to 0.00016098, passing the original bounds. No outcome/logit comparison was used to choose the remedy. Very tiny BF16 updates are quantized and need not track the intended direction closely; their actual direction cosines and energy are retained, including the low cosine of this case (~0.0613).

New namespace: /lambda/nfs/dami/hss/sink-next-20260925/exp4-v2.

The operator first executes the unchanged v3 implementation. Only if its actual-active-mask and per-token-norm gates pass but the relative energy gate fails does it retry the same computation with internal_absolute_norm_target=0. All outer tolerances, axis, threshold, control direction, questions, decoding, endpoints, tests and timing policy remain unchanged. No new random directions or discarded questions. Any further failure still stops execution.

All 50 successfully saved records and 10 zero replay checks are imported byte-for-byte with original hashes, provenance and new-plan receipts. The original successful operator output is unchanged by this fallback, so these remain compatible. The failed condition is regenerated from its original prompt and fresh cache. The new plan explicitly lists imported JSON/NPZ/replay hashes before resumption; no partial outcome is used to choose settings or stop the study. The already-disclosed partial counts remain exploratory progress, not the final comparison.

The original CPU auditor is reused under the new plan, checks every saved update against the unchanged gates, and reports only the complete 606-record cohort. The report publication adapter must identify the v2 namespace and amendment; v1 failure is never relabeled as success.
