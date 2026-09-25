# Online clamp generation: numerical amendment v3

The v2 run stopped after 128 passing records at math_2147/ORTH_MAN_1 on 2026-09-25 19:04:35 UTC. Its tiny target update norm was 3.6137968e-5; 32 rounding-repair iterations left relative energy error 0.00724723, above the unchanged 0.002 gate. Source records and the failed namespace remain immutable.

Exact replay saved the offending activation. Numerical-only checks (no logit or generated-answer comparisons) showed that 128 iterations give relative energy error 0.000892291 and absolute norm error 1.6127160e-8, passing the original gates. Budgets of 512 and 2048 produce the same result. This is iteration exhaustion, distinct from the earlier absolute-floor issue.

New namespace: `exp4-v3`. The solver first executes v2 unchanged and returns its output when it passes. Only on the existing relative-energy failure, with active mask and token norm gates passing, it retries the same deterministic rounding procedure with budgets 128, 512, then 2048. It returns the first passing result and still stops on unresolved failure. No skipped questions, relaxed gates, new axes, changed thresholds, decoding, endpoints or cohorts.

All 128 successful records and 10 replay checks are imported byte-for-byte with hashes and new-plan receipts. The failed condition restarts from its prompt. Preflight checks both saved historical failing activations and exact preservation of successful v2 output. As before, BF16 micro-updates need not align closely with the ideal direction (this case cosine about 0.04664); actual direction diagnostics remain recorded. No claim of exact realized orthogonality is made.

The 19:05 UTC answer-review snapshot includes 43 zero/C1 pairs, of which 42 also have the control. Review is separate, exploratory and cannot change the frozen trial or its scoring.
