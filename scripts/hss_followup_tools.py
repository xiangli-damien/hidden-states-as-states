"""
hss_followup_tools.py -- frozen helpers for the HSS sink-intervention follow-up.

Spec: HSS_followup_spec_v1.md (v1.0, 2026-09-25). Freeze this file together with the
spec: record the SHA-256 of both files and a timestamp BEFORE running Steps 1-3.
Any later change goes into deviations.md (time, what, why).

Sections
  1. Frozen parameters
  2. Text utilities      (cut point, forced-answer parsing, majority vote, seeds)
  3. Vector construction (split, d / sink axis, tau, type-matched targets, controls, chart vectors)
  4. Interventions       (numpy reference implementation + torch version for hooks)
  5. Statistics          (bootstrap CI, exact McNemar, paired binary summary)
  6. Step-2 decision rule
  7. Self-tests          (run:  python hss_followup_tools.py)

Only numpy / pandas / scipy are needed, except make_torch_transform and
check_torch_equivalence, which import torch lazily.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Frozen parameters (every value here is listed in spec section A)
# ---------------------------------------------------------------------------
PARAMS = dict(
    alpha=0.3,                    # same as the current run
    tau_percentile=75.0,          # C1 clamp threshold, percentile of non-sink token projections
    n_ctrl_iso=5,                 # isotropic controls per family
    n_ctrl_man=5,                 # on-manifold controls per family
    bootstrap_B=10_000,
    seed=20260925,
    collateral_abs_bound=-0.10,   # nats; mean delta M_col_ans of a candidate must be >= this
    tie_order=["C1", "C3", "C2", "C0"],   # among passing candidates statistically tied with the best
                                          # (paired bootstrap CI of the difference includes 0), prefer the
                                          # least invasive one in this order
    min_repeat_chars=20,          # a repeated sentence must have >= this many chars (normalised)
    min_loop_tokens=20,           # M_loop is missing if the continuation after the cut is shorter
    type_min_correct=20,          # type-matched target: minimum correct responses
    type_min_sink=10,             # type-matched target: minimum sink responses of that type
    tau_sample_responses=300,     # non-sink responses sampled for tau
    tau_max_tokens_per_response=200,
    force_string="\n\nTherefore, the final answer is $\\boxed{",  # CONFIRM format before freezing
    force_max_new_tokens=40,
    retry_k=4,
    retry_temperature=0.7,
    controls_fraction=0.95,
    hook_layer=14,
    chart_rank=8,
    projection_zero_relative=1e-6,
    normal_screen_questions=100,
    max_new_tokens=2048,
    freeze_utc="2026-09-26T01:59:00+00:00",
    finish_reserve_seconds=3600,
    identity_replay_questions=10,
    retry_top_p=1.0,
)
CANDIDATES = ["C0", "C1", "C2", "C3"]
NULL_FAMILY = {"C0": "CTRL_ADD", "C2": "CTRL_ADD", "C1": "CTRL_CLAMP", "C3": "CTRL_PROJ"}
CONTROL_FAMILIES = ["CTRL_ADD", "CTRL_CLAMP", "CTRL_PROJ"]
NONE_COND = "NONE"


def control_condition_names(family: str, params: dict = PARAMS) -> list[str]:
    """Naming convention used by step2_decision: {family}_{iso|man}_{k}, k starting at 1."""
    return ([f"{family}_iso_{k}" for k in range(1, params["n_ctrl_iso"] + 1)]
            + [f"{family}_man_{k}" for k in range(1, params["n_ctrl_man"] + 1)])


# ---------------------------------------------------------------------------
# 2. Text utilities
# ---------------------------------------------------------------------------
def normalize_sentence(s: str) -> str:
    """Whitespace-normalised sentence used for repeat detection."""
    return " ".join(s.split())


def cut_token_index(sent_texts, sent_tok_starts, n_tokens, min_repeat_chars=PARAMS["min_repeat_chars"]):
    """Cut point of a sink response for M_ans / M_loop (spec D5).

    sent_texts      sentences in order (same splitter as paper section 5.2)
    sent_tok_starts token index (relative to the response, first token = 0) where each sentence starts
    n_tokens        number of response tokens

    Returns (cut, rule) with 0 < cut < n_tokens, or (None, "empty") if n_tokens < 2.
      rule "first_repeat": start of the first sentence (>= min_repeat_chars after normalisation)
                            identical to an earlier sentence;
      rule "half_sentence": otherwise the sentence start closest to n_tokens/2 (ties -> earlier), excluding 0;
      rule "half_token":    otherwise n_tokens // 2.
    """
    if n_tokens < 2:
        return None, "empty"
    if len(sent_texts) != len(sent_tok_starts):
        raise ValueError("sent_texts and sent_tok_starts differ in length")
    seen = set()
    for text, start in zip(sent_texts, sent_tok_starts):
        key = normalize_sentence(text)
        if len(key) < min_repeat_chars:
            continue
        if key in seen and 0 < start < n_tokens:
            return int(start), "first_repeat"
        seen.add(key)
    inner = [int(s) for s in sent_tok_starts if 0 < s < n_tokens]
    if inner:
        half = n_tokens / 2.0
        return min(inner, key=lambda s: (abs(s - half), s)), "half_sentence"
    return n_tokens // 2, "half_token"


def extract_braced(text: str):
    """Content up to the brace that closes an ALREADY OPENED '{' (text starts right after '\\boxed{').

    Escaped characters (e.g. '\\{', '\\}') are skipped. Returns None if the brace never closes.
    """
    depth, i = 1, 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i]
        i += 1
    return None


def majority_vote(answers, normalize=lambda a: a):
    """Majority over k extracted boxed answers (None / empty = no answer).

    `normalize` must be the frozen scorer's answer normaliser. Ties -> earliest sample.
    Returns the original (un-normalised) answer string of the winner, or None.
    """
    norm = [None if (a is None or str(a).strip() == "") else normalize(a) for a in answers]
    norm = [None if n is None or str(n).strip() == "" else n for n in norm]
    counts = Counter(n for n in norm if n is not None)
    if not counts:
        return None
    top = max(counts.values())
    for a, n in zip(answers, norm):
        if n is not None and counts[n] == top:
            return a
    return None


def stable_int(s: str) -> int:
    """Process-independent integer hash (Python's hash() is salted per process)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def sample_seed(qid: str, arm: str, j: int, seed: int = PARAMS["seed"]) -> int:
    """Seed for the j-th sampled continuation of question qid in arm (Step 3B)."""
    return (stable_int(f"{qid}|{arm}|{j}") ^ seed) & 0x7FFFFFFF


def retry_prefix_boundaries(n_sentences: int, f: int, qid: str, seed: int = PARAMS["seed"]):
    """Step 3B prefixes, as numbers of sentences kept.

    Boundaries: b_0 = start, b_i = end of sentence i (i = 1..n). f = index of the first alarm,
    must be non-final (1 <= f < n). R-HSS keeps f-1 sentences; R-RAND keeps u ~ Uniform{0..n-1}
    (fixed per question); R-START keeps 0.
    """
    if not (1 <= f < n_sentences):
        raise ValueError("f must satisfy 1 <= f < n_sentences")
    rng = np.random.default_rng([seed, stable_int(qid)])
    return dict(hss=f - 1, rand=int(rng.integers(0, n_sentences)), start=0)


# ---------------------------------------------------------------------------
# 3. Vector construction (all quantities at the block-14 hook point, A split only)
# ---------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("cannot normalise a zero / non-finite vector")
    return v / n


def stratified_half_split(ids, strata, seed: int = PARAMS["seed"]):
    """Stratified 50/50 split of question ids. Within each stratum ids are shuffled and dealt
    alternately to A and B starting with A (so singletons go to A). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": list(ids), "s": [str(x) for x in strata]})
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids")
    A, B = [], []
    for s in sorted(df["s"].unique()):
        g = df.loc[df["s"] == s, "id"].to_numpy()
        perm = list(g[rng.permutation(len(g))])
        A.extend(perm[0::2])
        B.extend(perm[1::2])
    return set(A), set(B)


def sink_direction(means_nonsink_A, means_sink_A):
    """d = mean(non-sink) - mean(sink); s_hat = unit vector pointing TO the sink (= -d/|d|)."""
    mu_non = np.asarray(means_nonsink_A, dtype=np.float64).mean(axis=0)
    mu_sink = np.asarray(means_sink_A, dtype=np.float64).mean(axis=0)
    d = mu_non - mu_sink
    return d, -unit(d)


def subsample_token_indices(n_tokens: int, rng, max_per=PARAMS["tau_max_tokens_per_response"]):
    """Uniform subsample (without replacement) of generated-token indices of one response."""
    if n_tokens <= max_per:
        return np.arange(n_tokens)
    return np.sort(rng.choice(n_tokens, size=max_per, replace=False))


def tau_from_projections(z, pct=PARAMS["tau_percentile"]):
    """Clamp threshold: percentile of token projections z = h . s_hat (non-sink A tokens)."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        raise ValueError("no projections")
    return float(np.percentile(z, pct))


def type_matched_targets(meta_A: pd.DataFrame, means_A, sink_types, d_norm,
                         min_correct=PARAMS["type_min_correct"], min_sink=PARAMS["type_min_sink"]):
    """C2 vectors v_T with |v_T| = d_norm, one per problem type present in S_B.

    meta_A: one row per A-split response, row i aligned with means_A[i];
            columns: type (str), level (int), correct (0/1), is_sink (bool).
    Target tiers (first with >= min_correct responses wins): same type & correct & level>=4;
    level>=3; any level; else all correct non-sink responses (logged as fallback).
    Sink side: same-type sink responses, or all sink responses if fewer than min_sink.
    """
    means_A = np.asarray(means_A, dtype=np.float64)
    if len(meta_A) != means_A.shape[0]:
        raise ValueError("meta_A and means_A are not aligned")
    t = meta_A["type"].astype(str).to_numpy()
    lv = meta_A["level"].to_numpy()
    c = meta_A["correct"].to_numpy().astype(bool)
    s = meta_A["is_sink"].to_numpy().astype(bool)
    out = {}
    for T in sorted({str(x) for x in sink_types}):
        tiers = [("level>=4", (~s) & (t == T) & c & (lv >= 4)),
                 ("level>=3", (~s) & (t == T) & c & (lv >= 3)),
                 ("any_level", (~s) & (t == T) & c),
                 ("all_types_correct", (~s) & c)]
        name, mask = next(((n, m) for n, m in tiers if m.sum() >= min_correct), tiers[-1])
        smask, srule = s & (t == T), "same_type"
        if smask.sum() < min_sink:
            smask, srule = s, "all_sink"
        dT = means_A[mask].mean(axis=0) - means_A[smask].mean(axis=0)
        out[T] = dict(v=d_norm * unit(dT), target_rule=name, n_target=int(mask.sum()),
                      sink_rule=srule, n_sink=int(smask.sum()))
    return out


def iso_controls(D: int, k: int = PARAMS["n_ctrl_iso"], seed: int = PARAMS["seed"] + 1):
    """k isotropic random unit vectors in R^D (rows)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((k, D))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def manifold_controls(centroids_nonsink, k: int = PARAMS["n_ctrl_man"], seed: int = PARAMS["seed"] + 2):
    """k unit vectors +/-(c_i - c_j) for distinct random pairs of NON-sink response-level centroids
    at the hook layer. Returns (V, pairs)."""
    C = np.asarray(centroids_nonsink, dtype=np.float64)
    S = C.shape[0]
    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    if len(pairs) < k:
        raise ValueError(f"only {len(pairs)} centroid pairs for {k} controls")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    chosen = [pairs[p] for p in idx]
    V = np.stack([sg * (C[i] - C[j]) for sg, (i, j) in zip(signs, chosen)])
    return np.stack([unit(v) for v in V]), chosen


def chart_vectors(bases, v):
    """C3 lookup table. bases: (S, D, r) orthonormal local bases of the token-level map; v: (D,).

    Returns (U, ratio): U[s] = unit(U_s U_s^T v) (a zero row where the projection vanishes, i.e.
    no intervention for tokens in that region), ratio[s] = |U_s U_s^T v| / |v|.
    Because v is fixed, the projected direction depends only on the region, so it is precomputed.
    """
    bases = np.asarray(bases, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    unit(v)  # Fail closed on zero/nonfinite directions.
    coeff = np.einsum("sdr,d->sr", bases, v)
    proj = np.einsum("sdr,sr->sd", bases, coeff)
    norms = np.linalg.norm(proj, axis=1)
    ok = norms > PARAMS["projection_zero_relative"] * np.linalg.norm(v)
    U = np.zeros_like(proj)
    U[ok] = proj[ok] / norms[ok, None]
    return U, norms / np.linalg.norm(v)


def nearest_centroid(h, centroids):
    """Index of the nearest centroid (Euclidean) for each row of h. Must match the token-level
    map's assignment rule; replace if that map assigns by GMM posterior instead."""
    h = np.asarray(h, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    d2 = (h * h).sum(1, keepdims=True) - 2.0 * h @ C.T + (C * C).sum(1)[None, :]
    return d2.argmin(axis=1)


# ---------------------------------------------------------------------------
# 4. Interventions. h: (n_tokens, D) activations at GENERATED positions only.
#    For C1 and C3 the rule is evaluated on the pre-intervention h.
# ---------------------------------------------------------------------------
def apply_add(h, v_unit, a):
    """C0 / C2 / CTRL_ADD: h <- h + a * v_unit."""
    return np.asarray(h, dtype=np.float64) + a * np.asarray(v_unit, dtype=np.float64)[None, :]


def apply_clamp(h, s_hat, tau):
    """C1 / CTRL_CLAMP: z = h.s_hat; where z > tau, remove the excess (z - tau) along s_hat."""
    h = np.asarray(h, dtype=np.float64)
    s_hat = np.asarray(s_hat, dtype=np.float64)
    z = h @ s_hat
    excess = np.maximum(z - tau, 0.0)
    return h - excess[:, None] * s_hat[None, :]


def apply_chart(h, chart_units, a, centroids_tok):
    """C3 / CTRL_PROJ: h <- h + a * U[s(h)], s(h) = token-level region of the pre-intervention h."""
    h = np.asarray(h, dtype=np.float64)
    s = nearest_centroid(h, centroids_tok)
    return h + a * np.asarray(chart_units, dtype=np.float64)[s]


def make_torch_transform(kind: str, *, a=None, v_unit=None, s_hat=None, tau=None,
                         chart_units=None, centroids_tok=None, device=None):
    """torch version for the existing block-14 hook. Returns f(h) for h[..., D].

    Computes in float32 and casts back to h.dtype. Use inside the hook on generated positions only:
        hs[gen_mask] = f(hs[gen_mask])
    kind: "none" | "add" | "clamp" | "chart".  Verify with check_torch_equivalence() first.
    """
    import torch

    def T(x):
        return None if x is None else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    v, sh, cu, ct = T(v_unit), T(s_hat), T(chart_units), T(centroids_tok)
    ct_sq = None if ct is None else (ct * ct).sum(-1)

    def f(h):
        dt = h.dtype
        x = h.float()
        if kind == "none":
            y = x
        elif kind == "add":
            y = x + a * v
        elif kind == "clamp":
            z = x @ sh
            y = x - torch.clamp(z - tau, min=0.0).unsqueeze(-1) * sh
        elif kind == "chart":
            d2 = (x * x).sum(-1, keepdim=True) - 2.0 * (x @ ct.T) + ct_sq
            y = x + a * cu[d2.argmin(-1)]
        else:
            raise ValueError(kind)
        return y.to(dt)

    return f


def check_torch_equivalence(D=64, n=200, S=7, r=4, seed=0, device=None):
    """Run once on the GPU machine: max |torch - numpy| per intervention kind (expect < 1e-4)."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, D)).astype(np.float32)
    v = unit(rng.standard_normal(D))
    s_hat = unit(rng.standard_normal(D))
    tau = float(np.percentile(h @ s_hat, 75))
    bases = np.stack([np.linalg.qr(rng.standard_normal((D, r)))[0] for _ in range(S)])
    cents = rng.standard_normal((S, D))
    cu, _ = chart_vectors(bases, v)
    import torch
    ht = torch.as_tensor(h, device=device)
    out = {}
    for kind, ref, kw in [
        ("add", apply_add(h, v, 0.7), dict(a=0.7, v_unit=v)),
        ("clamp", apply_clamp(h, s_hat, tau), dict(s_hat=s_hat, tau=tau)),
        ("chart", apply_chart(h, cu, 0.7, cents), dict(a=0.7, chart_units=cu, centroids_tok=cents)),
    ]:
        got = make_torch_transform(kind, device=device, **kw)(ht).cpu().numpy()
        out[kind] = float(np.abs(got - ref).max())
    return out


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(x, B=PARAMS["bootstrap_B"], seed=PARAMS["seed"], level=0.95):
    """Mean and percentile bootstrap CI (resampling units = questions)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("empty or nonfinite sample")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(x.mean()), float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    from scipy import stats  # Only the CPU statistics stage needs scipy.
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    return float(stats.binomtest(int(b), n, 0.5, alternative="two-sided").pvalue)


def paired_binary_summary(success_a, success_b) -> dict:
    """Paired comparison of two arms on a binary endpoint (e.g. correct AND boxed)."""
    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("arms are not paired")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return dict(n=int(a.size), a_success=int(a.sum()), b_success=int(b.sum()),
                only_a=only_a, only_b=only_b, both=int((a & b).sum()),
                neither=int((~a & ~b).sum()), mcnemar_p=mcnemar_exact(only_a, only_b))


# ---------------------------------------------------------------------------
# 6. Step-2 decision rule (spec D6). Do not decide by hand.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["question_id", "set", "condition", "family", "m_ans", "m_loop", "m_col_ans", "m_col_nll"]


def _deltas(screen, metric, qset, conds):
    sub = screen[(screen["set"] == qset) & (screen["condition"].isin(conds + [NONE_COND]))]
    wide = sub.pivot(index="question_id", columns="condition", values=metric)
    missing = [c for c in conds + [NONE_COND] if c not in wide.columns]
    if missing:
        raise ValueError(f"conditions missing for {metric}/{qset}: {missing}")
    wide = wide[conds + [NONE_COND]]
    if np.isinf(wide.to_numpy()).any():
        raise ValueError(f"infinite {metric}/{qset}")
    if metric != "m_loop" and (wide.empty or wide.isna().any().any()):
        raise ValueError(f"missing mandatory paired values: {metric}/{qset}")
    wide = wide.dropna()   # Only the prespecified short-continuation metric may be missing.
    return wide[conds].sub(wide[NONE_COND], axis=0), int(len(wide))


def step2_decision(screen: pd.DataFrame, c0_generation_beats_random: bool, params: dict = PARAMS) -> dict:
    """Apply P1-P4 and the decision of spec D6.

    screen: long table, one row per (question, condition), columns REQUIRED_COLUMNS:
      set       "sink" (S_B) or "normal" (M_B)
      condition "NONE", "C0".."C3", or control names from control_condition_names()
      family    "NONE", "C0".."C3", "CTRL_ADD", "CTRL_CLAMP", "CTRL_PROJ"
      m_ans, m_loop       sink rows (NaN elsewhere); m_col_ans, m_col_nll  normal rows (NaN elsewhere)
    c0_generation_beats_random: in the COMPLETE current run, did the HSS arm have more target
      questions that are correct AND boxed than the random arm? (strictly more -> True)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in screen.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if screen.duplicated(["question_id", "condition"]).any():
        raise ValueError("duplicate (question_id, condition) rows")
    fam_of = screen.groupby("condition")["family"].agg(lambda s: set(s))
    if any(len(v) != 1 for v in fam_of):
        raise ValueError("a condition is assigned to more than one family")
    fam_of = fam_of.map(lambda s: next(iter(s)))
    if fam_of.get(NONE_COND) != "NONE":
        raise ValueError("condition NONE (family NONE) is required")
    expected_all = {NONE_COND, *CANDIDATES}
    for fam in CONTROL_FAMILIES:
        expected_all.update(control_condition_names(fam, params))
    if set(fam_of.index) != expected_all:
        raise ValueError("unexpected or missing condition")
    for C in CANDIDATES:
        if fam_of.get(C) != C:
            raise ValueError(f"wrong family for {C}")
    for qset in ["sink", "normal"]:
        groups = screen[screen['set'] == qset].groupby('question_id')['condition'].agg(set)
        if groups.empty or any(g != expected_all for g in groups):
            raise ValueError(f"incomplete question-condition grid: {qset}")
    for fam in CONTROL_FAMILIES:
        expected = control_condition_names(fam, params)
        found = sorted(fam_of[fam_of == fam].index)
        if sorted(expected) != found:
            raise ValueError(f"{fam}: expected controls {expected}, found {found}")

    B, seed = params["bootstrap_B"], params["seed"]
    per, per_q_d_ans = {}, {}
    for C in CANDIDATES:
        if C not in fam_of.index:
            raise ValueError(f"candidate {C} missing")
        ctrl = control_condition_names(NULL_FAMILY[C], params)
        conds = [C] + ctrl
        d_ans, n_ans = _deltas(screen, "m_ans", "sink", conds)
        d_loop, n_loop = _deltas(screen, "m_loop", "sink", conds)
        d_col, n_col = _deltas(screen, "m_col_ans", "normal", conds)
        d_nll, n_nll = _deltas(screen, "m_col_nll", "normal", conds)

        per_q_d_ans[C] = d_ans[C]
        mean_c, lo, hi = bootstrap_mean_ci(d_ans[C].to_numpy(), B=B, seed=seed)
        ctrl_means = d_ans[ctrl].mean(axis=0)
        n_exceeded = int((mean_c > ctrl_means).sum())
        need = math.ceil(params["controls_fraction"] * len(ctrl))
        col_c = float(d_col[C].mean())
        col_ctrl = d_col[ctrl].mean(axis=0)

        p1 = lo > 0
        p2 = n_exceeded >= need
        p3 = (col_c >= float(col_ctrl.min())) and (col_c >= params["collateral_abs_bound"])
        p4 = float(d_loop[C].mean()) > 0
        per[C] = dict(
            null_family=NULL_FAMILY[C], n_sink=n_ans, n_loop=n_loop, n_normal=n_col, n_normal_nll=n_nll,
            mean_d_ans=mean_c, ci95_d_ans=[lo, hi],
            ctrl_mean_d_ans={k: float(v) for k, v in ctrl_means.items()},
            n_controls_exceeded=n_exceeded, n_controls_needed=need,
            mean_d_loop=float(d_loop[C].mean()) if n_loop else None,
            mean_d_col_ans=col_c, ctrl_min_d_col_ans=float(col_ctrl.min()),
            ctrl_median_d_col_ans=float(col_ctrl.median()),
            mean_d_col_nll=float(d_nll[C].mean()),
            P1_ci_above_zero=bool(p1), P2_beats_controls=bool(p2),
            P3_collateral_ok=bool(p3), P4_loop_up_reported=bool(p4),
            passed=bool(p1 and p2 and p3),
        )

    pass_set = [C for C in CANDIDATES if per[C]["passed"]]
    miscalibrated = ("C0" in pass_set) and (not c0_generation_beats_random)
    new_pass = [C for C in pass_set if C != "C0"]
    tie_ci = {}          # candidate -> 95% CI of (best - candidate) per-question delta M_ans
    if miscalibrated:
        step3, chosen = "3B", None
        reason = "C0 passed the screen but did not beat random in generation: screen judged not predictive"
    elif new_pass:
        best = max(new_pass, key=lambda C: per[C]["mean_d_ans"])
        tied = [best]
        for C in new_pass:
            if C == best:
                continue
            common = per_q_d_ans[best].index.intersection(per_q_d_ans[C].index)
            diff = (per_q_d_ans[best].loc[common] - per_q_d_ans[C].loc[common]).to_numpy()
            _, lo_d, hi_d = bootstrap_mean_ci(diff, B=B, seed=seed)
            tie_ci[C] = [lo_d, hi_d]
            if lo_d <= 0 <= hi_d:               # The interval must actually contain zero.
                tied.append(C)
        chosen = next(C for C in params["tie_order"] if C in tied)
        step3 = "3A"
        reason = (f"{chosen} passed P1-P3; best mean = {best}; tied with best (paired CI incl. 0): "
                  f"{sorted(tied)}; chosen by preference order {params['tie_order']}")
    else:
        step3, chosen = "3B", None
        reason = "no new candidate (C1-C3) passed P1-P3"
    return dict(
        spec_version="1.1-reviewed",
        params={k: v for k, v in params.items()},
        c0_generation_beats_random=bool(c0_generation_beats_random),
        per_candidate=per, pass_set=pass_set, screen_miscalibrated=bool(miscalibrated),
        tie_ci_best_minus_candidate=tie_ci,
        step3=step3, chosen=chosen,
        matched_control_for_3A=(f"{NULL_FAMILY[chosen]}_man_1" if chosen else None),
        reason=reason,
    )


def decision_table_markdown(decision: dict) -> str:
    """Human-readable summary of step2_decision output."""
    lines = ["| cand | null | n | mean dAns [95% CI] | beats ctrl | mean dLoop | mean dColAns (ctrl min) | P1 | P2 | P3 | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for C, r in decision["per_candidate"].items():
        lines.append(
            f"| {C} | {r['null_family']} | {r['n_sink']} | {r['mean_d_ans']:.3f} "
            f"[{r['ci95_d_ans'][0]:.3f}, {r['ci95_d_ans'][1]:.3f}] | {r['n_controls_exceeded']}/{r['n_controls_needed']} "
            f"| {r['mean_d_loop']} | {r['mean_d_col_ans']:.3f} ({r['ctrl_min_d_col_ans']:.3f}) "
            f"| {r['P1_ci_above_zero']} | {r['P2_beats_controls']} | {r['P3_collateral_ok']} | {r['passed']} |")
    lines.append(f"\nstep3 = {decision['step3']}, chosen = {decision['chosen']}, "
                 f"matched control = {decision['matched_control_for_3A']}. {decision['reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Self-tests
# ---------------------------------------------------------------------------
def _synthetic_screen(offsets_ans=None, offsets_col=None, n_sink=98, n_norm=100, seed=1, params=PARAMS):
    rng = np.random.default_rng(seed)
    conds = [(NONE_COND, "NONE")] + [(c, c) for c in CANDIDATES]
    for fam in CONTROL_FAMILIES:
        conds += [(name, fam) for name in control_condition_names(fam, params)]
    base_ans = rng.normal(-8.0, 2.0, n_sink)
    base_loop = rng.normal(0.5, 0.2, n_sink)
    base_col = rng.normal(-0.05, 0.02, n_norm)
    base_nll = rng.normal(0.4, 0.1, n_norm)
    rows = []
    for cond, fam in conds:
        noise = 0.0 if cond == NONE_COND else 1.0
        oa = (offsets_ans or {}).get(cond, 0.0)
        oc = (offsets_col or {}).get(cond, 0.0)
        ans = base_ans + oa + noise * rng.normal(0, 0.3, n_sink)
        loop = base_loop + noise * rng.normal(0, 0.05, n_sink)
        col = base_col + oc + noise * rng.normal(0, 0.02, n_norm)
        nll = base_nll + noise * rng.normal(0, 0.01, n_norm)
        for i in range(n_sink):
            rows.append(dict(question_id=f"s{i}", set="sink", condition=cond, family=fam,
                             m_ans=ans[i], m_loop=loop[i], m_col_ans=np.nan, m_col_nll=np.nan))
        for i in range(n_norm):
            rows.append(dict(question_id=f"n{i}", set="normal", condition=cond, family=fam,
                             m_ans=np.nan, m_loop=np.nan, m_col_ans=col[i], m_col_nll=nll[i]))
    return pd.DataFrame(rows)


def _self_test():
    # text utilities
    long1 = "Let x be the number of apples in the basket today."
    long2 = "Then 2x + 3 = 11, so x = 4 apples in total here."
    assert cut_token_index([long1, long2, long1, long2], [0, 12, 25, 38], 50) == (25, "first_repeat")
    assert cut_token_index(["So,", "So,", long1, long2], [0, 2, 4, 30], 60) == (30, "half_sentence")
    assert cut_token_index([long1], [0], 9) == (4, "half_token")
    assert cut_token_index([], [], 1) == (None, "empty")
    assert extract_braced("\\frac{1}{2}} is the answer") == "\\frac{1}{2}"
    assert extract_braced("\\{1,2\\}}$") == "\\{1,2\\}"
    assert extract_braced("12") is None
    assert majority_vote(["3", "5", "3", None]) == "3"
    assert majority_vote(["5", "3", None, ""]) == "5"
    assert majority_vote([None, None]) is None
    b = retry_prefix_boundaries(10, 4, "math_1")
    assert b["hss"] == 3 and b["start"] == 0 and 0 <= b["rand"] <= 9
    assert b == retry_prefix_boundaries(10, 4, "math_1")          # deterministic per question
    assert sample_seed("q", "R-HSS", 0) == sample_seed("q", "R-HSS", 0)

    # vectors and interventions
    rng = np.random.default_rng(0)
    D, S, r = 32, 6, 4
    h = rng.standard_normal((500, D))
    s_hat = unit(rng.standard_normal(D))
    tau = tau_from_projections(h @ s_hat)
    out = apply_clamp(h, s_hat, tau)
    z0, z1 = h @ s_hat, out @ s_hat
    below = z0 <= tau
    assert np.allclose(out[below], h[below])                         # untouched below tau
    assert np.allclose(z1[~below], tau)                              # clamped exactly to tau
    perp0 = h - np.outer(z0, s_hat)
    perp1 = out - np.outer(z1, s_hat)
    assert np.allclose(perp0, perp1)                                 # orthogonal part unchanged
    assert abs(below.mean() - 0.75) < 0.01
    bases = np.stack([np.linalg.qr(rng.standard_normal((D, r)))[0] for _ in range(S)])
    v = rng.standard_normal(D)
    U, ratio = chart_vectors(bases, v)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0) and np.all((ratio > 0) & (ratio <= 1 + 1e-12))
    cents = rng.standard_normal((S, D)) * 3
    hc = apply_chart(h, U, 0.5, cents)
    s_idx = nearest_centroid(h, cents)
    assert np.allclose(hc - h, 0.5 * U[s_idx])
    assert np.allclose(apply_add(h, unit(v), 2.0) - h, 2.0 * unit(v))
    V, pairs = manifold_controls(cents, k=5, seed=3)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0) and len(set(pairs)) == 5
    Ri = iso_controls(D, 5)
    assert np.allclose(np.linalg.norm(Ri, axis=1), 1.0)

    # split
    ids = [f"q{i}" for i in range(21)]
    strata = ["alg"] * 10 + ["geo"] * 7 + ["nt"] * 3 + ["prob"]
    A, Bs = stratified_half_split(ids, strata, seed=5)
    assert A.isdisjoint(Bs) and len(A | Bs) == 21 and "q20" in A     # singleton goes to A

    # type-matched targets with fallbacks
    n = 200
    meta = pd.DataFrame({
        "type": ["IA"] * 100 + ["Geo"] * 95 + ["NT"] * 5,
        "level": list(rng.integers(1, 6, 100)) + [2] * 95 + [5] * 5,
        "correct": list(rng.integers(0, 2, 100)) + [1] * 95 + [0] * 5,
        "is_sink": [True] * 30 + [False] * 70 + [False] * 90 + [True] * 5 + [True] * 5,
    })
    means = rng.standard_normal((n, D))
    tm = type_matched_targets(meta, means, ["IA", "Geo", "NT"], d_norm=7.0)
    for T, info in tm.items():
        assert abs(np.linalg.norm(info["v"]) - 7.0) < 1e-9
    assert tm["Geo"]["target_rule"] == "any_level" and tm["Geo"]["sink_rule"] == "all_sink"
    assert tm["NT"]["target_rule"] == "all_types_correct"

    # statistics
    m, lo, hi = bootstrap_mean_ci(np.ones(50) * 0.2 + rng.normal(0, 0.01, 50), B=2000)
    assert lo < m < hi and lo > 0.19
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(10, 0) < 0.01
    pb = paired_binary_summary([1, 1, 0, 0, 1], [1, 0, 0, 1, 0])
    assert (pb["only_a"], pb["only_b"], pb["both"], pb["neither"]) == (2, 1, 1, 1)

    # decision branches
    def run(oa=None, oc=None, c0_gen=False):
        return step2_decision(_synthetic_screen(oa, oc), c0_generation_beats_random=c0_gen)

    d = run({"C1": 0.5})
    assert d["step3"] == "3A" and d["chosen"] == "C1" and d["matched_control_for_3A"] == "CTRL_CLAMP_man_1", d
    d = run()
    assert d["step3"] == "3B" and d["chosen"] is None, d
    d = run({"C0": 0.5})
    assert d["screen_miscalibrated"] and d["step3"] == "3B", d
    d = run({"C1": 0.5}, {"C1": -0.5})
    assert d["step3"] == "3B" and not d["per_candidate"]["C1"]["P3_collateral_ok"], d
    d = run({"C1": 0.5, "C3": 0.5})
    assert d["chosen"] == "C1", d                                      # tie -> C1 first
    d = run({"C1": 0.4, "C3": 0.6})
    assert d["chosen"] == "C3" and d["matched_control_for_3A"] == "CTRL_PROJ_man_1", d
    d = run({"C0": 0.5, "C2": 0.5}, c0_gen=True)
    assert d["step3"] == "3A" and d["chosen"] == "C2", d               # C0 consistent, C0 never goes to 3A
    json.dumps(d)                                                       # serialisable
    return decision_table_markdown(run({"C1": 0.5}))


if __name__ == "__main__":
    print(_self_test())
    print("\nall self-tests passed")
