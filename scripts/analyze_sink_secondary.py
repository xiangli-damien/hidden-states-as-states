"""Frozen CPU-only secondary analysis of sealed sink-intervention exports."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sink_secondary_20260925.json"
EXPECTED_CONFIG = "13aee7ee640295d88caed353454658236d5829b6e5ea600215dfe8ec42c5898f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def exact_sign(wins, losses):
    return float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.


def conditional_exact(states, behaviors):
    """Condition on both margins per question; enumerate the null by convolution.

    With 3 arms the three pairwise signs are dependent, and even a majority sign
    per question need not have probability 1/2 under fixed margins. Hypergeometric
    overlap is the appropriate exact conditional null for this score statistic.
    """
    states, behaviors = np.asarray(states, int), np.asarray(behaviors, int)
    n = states.shape[1]
    dist = np.array([1.])
    observed_overlap = int((states * behaviors).sum())
    offset = 0
    for s, d in zip(states, behaviors):
        a, b = int(s.sum()), int(d.sum())
        offset += a * b
        pmf = hypergeom.pmf(np.arange(n + 1), n, a, b)
        dist = np.convolve(dist, pmf)
    dist /= dist.sum()
    lower, upper = dist[:observed_overlap + 1].sum(), dist[observed_overlap:].sum()
    return dict(n_questions=len(states), statistic=int(n * observed_overlap - offset),
                overlap=observed_overlap, p_two_sided=float(min(1, 2 * min(lower, upper))),
                lower_tail=float(lower), upper_tail=float(upper),
                test="exact within-question conditional permutation, doubled smaller inclusive tail")


def bootstrap_means(x, B, seed):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    assert np.isfinite(x).all() and len(x) > 0
    rng = np.random.default_rng(seed)
    out = np.empty((B, x.shape[1]))
    for start in range(0, B, 500):
        weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=min(500, B-start))
        out[start:start + len(weights)] = weights @ x / len(x)
    return out


def ci(values, level):
    return np.quantile(values, [(1-level)/2, (1+level)/2]).tolist()


def effect(values, samples, level=.9875):
    return dict(mean=float(np.mean(values)), ci=ci(samples, level), ci_level=level, n=len(values))


def self_check():
    # Brute-force permutations independently verify the conditional test on a toy.
    s = np.array([[1, 0, 0], [1, 1, 0]])
    d = np.array([[1, 0, 0], [0, 1, 1]])
    overlaps = []
    for a, b in itertools.product(itertools.permutations(d[0]), itertools.permutations(d[1])):
        overlaps.append(int((s * np.array([a, b])).sum()))
    observed = int((s * d).sum())
    expected = min(1, 2 * min(np.mean(np.array(overlaps) <= observed), np.mean(np.array(overlaps) >= observed)))
    assert abs(conditional_exact(s, d)["p_two_sided"] - expected) < 1e-12
    assert conditional_exact(np.zeros((2, 3)), d)["p_two_sided"] == 1
    assert exact_sign(10, 4) == binomtest(10, 14, .5).pvalue
    b = bootstrap_means(np.ones((5, 2)), 100, 1)
    np.testing.assert_array_equal(b, np.ones((100, 2)))


def behavior_analysis(frame, cfg, B, seed, out):
    arms = cfg["arms"]
    frame = frame[frame.arm.isin(arms)].copy()
    assert len(frame) == 465 and not frame.duplicated(["sample_id", "arm"]).any()
    assert set(frame.n_tokens.le(2048)) == {True}
    frame["sink"] = frame.global_ids.map(lambda x: json.loads(x)[cfg["hidden_state_index"]] == cfg["sink_global_id"])
    frame["degenerate"] = ~frame.boxed | (frame.n_tokens >= 2048)
    arrays = {k: frame.pivot(index="sample_id", columns="arm", values=k).reindex(columns=arms) for k in ["sink", "degenerate", "boxed", "correct", "n_tokens"]}
    assert all(not x.isna().any().any() for x in arrays.values())
    ids = arrays["sink"].index
    cohorts = frame.drop_duplicates("sample_id").set_index("sample_id").loc[ids, "split"].to_numpy()
    s, d = arrays["sink"].to_numpy(int), arrays["degenerate"].to_numpy(int)
    pairs, per_q = [], np.zeros((len(ids), 3), int)
    for qi, sid in enumerate(ids):
        for a, b in itertools.combinations(range(3), 2):
            sd, dd = s[qi, b] - s[qi, a], d[qi, b] - d[qi, a]
            label = "concordant" if sd*dd > 0 else "opposite" if sd*dd < 0 else "behavior_unchanged" if sd != 0 else "same_state"
            if sd:
                per_q[qi, {"concordant": 0, "opposite": 1, "behavior_unchanged": 2}[label]] += 1
            pairs.append(dict(sample_id=sid, cohort=cohorts[qi], arm_a=arms[a], arm_b=arms[b],
                              sink_a=int(s[qi, a]), sink_b=int(s[qi, b]),
                              degenerate_a=int(d[qi, a]), degenerate_b=int(d[qi, b]), classification=label))
    write(out / "behavior_pairs.json", pairs)
    def aggregate(mask):
        c, o, t = per_q[mask].sum(0).tolist()
        return dict(n_questions=int(mask.sum()), sink_discordant_pairs=c+o+t,
                    concordant=c, opposite=o, unchanged_behavior=t,
                    informative_questions=int((per_q[mask, :2].sum(1) > 0).sum()),
                    state_changing_questions=int((per_q[mask].sum(1) > 0).sum()),
                    association_test=conditional_exact(s[mask], d[mask]))
    pooled = aggregate(np.ones(len(ids), bool))
    boot = sum(bootstrap_means(per_q[cohorts == c], B, seed + i) * int((cohorts == c).sum()) for i, c in enumerate(cfg["cohorts"]))
    for name, denom in [("among_jointly_discordant", boot[:, :2].sum(1)), ("among_all_sink_discordant", boot.sum(1))]:
        denom0 = pooled["concordant"] + pooled["opposite"] if name == "among_jointly_discordant" else pooled["sink_discordant_pairs"]
        pooled[name] = dict(proportion=pooled["concordant"] / denom0 if denom0 else None,
                            ci95=ci(boot[denom > 0, 0] / denom[denom > 0], .95),
                            bootstrap_unit="question, stratified by original cohort")
    pair_results = []
    for a, b in itertools.combinations(range(3), 2):
        prod = (s[:, b]-s[:, a]) * (d[:, b]-d[:, a])
        c, o = int((prod > 0).sum()), int((prod < 0).sum())
        p = exact_sign(c, o)
        pair_results.append(dict(arms=[arms[a], arms[b]], concordant=c, opposite=o,
                                 sink_discordant=int((s[:, a] != s[:, b]).sum()),
                                 p_two_sided=p, p_bonferroni3=min(1, 3*p)))
    movements = {}
    for group in ["all", *cfg["cohorts"]]:
        mask = np.ones(len(ids), bool) if group == "all" else cohorts == group
        movements[group] = {}
        for k, arm in enumerate(arms[1:], 1):
            exit_mask = mask & (s[:, 0] == 1) & (s[:, k] == 0)
            entry_mask = mask & (s[:, 0] == 0) & (s[:, k] == 1)
            boxes, correct = arrays["boxed"].to_numpy(bool), arrays["correct"].to_numpy(bool)
            n_exit, n_entry = int(exit_mask.sum()), int(entry_mask.sum())
            delta = (s[:, 0] - s[:, k])[mask]
            movements[group][arm] = dict(n=int(mask.sum()), zero_in_sink=int(s[mask, 0].sum()),
                treated_in_sink=int(s[mask, k].sum()), exits=n_exit, entries=n_entry,
                net_sink_reduction=effect(delta, bootstrap_means(delta, B, seed+20+k)[:, 0], .95),
                p_mcnemar_two_sided=exact_sign(n_exit, n_entry),
                exit_recovery=dict(newly_boxed=int((exit_mask & ~boxes[:, 0] & boxes[:, k]).sum()),
                    kept_boxed=int((exit_mask & boxes[:, 0] & boxes[:, k]).sum()),
                    lost_boxed=int((exit_mask & boxes[:, 0] & ~boxes[:, k]).sum()),
                    remained_unboxed=int((exit_mask & ~boxes[:, 0] & ~boxes[:, k]).sum()),
                    degenerate_to_nondegenerate=int((exit_mask & (d[:, 0] == 1) & (d[:, k] == 0)).sum()),
                    correct_repairs=int((exit_mask & ~correct[:, 0] & correct[:, k]).sum()),
                    correct_damages=int((exit_mask & correct[:, 0] & ~correct[:, k]).sum())),
                exit_ids=ids[exit_mask].tolist(), entry_ids=ids[entry_mask].tolist())
    raw_counts = []
    for (cohort, arm), sub in frame.groupby(["split", "arm"]):
        raw_counts.append(dict(cohort=cohort, arm=arm, n=len(sub), sink=int(sub.sink.sum()),
                               degenerate=int(sub.degenerate.sum()), boxed=int(sub.boxed.sum()),
                               cap_reached=int((sub.n_tokens >= 2048).sum())))
    return dict(pooled=pooled, cohorts={c: aggregate(cohorts == c) for c in cfg["cohorts"]},
                pair_specific=pair_results, movements=movements, raw_counts=raw_counts)


def likelihood_analysis(frame, cfg, B, seed):
    assert len(frame) == 6895 and not frame.duplicated(["question_id", "condition"]).any()
    sink = frame[frame["set"] == "sink"].pivot(index="question_id", columns="condition", values="m_loop")
    normal = frame[frame["set"] == "normal"].pivot(index="question_id", columns="condition", values="m_col_nll")
    assert len(sink) == 97 and len(normal) == 100
    rules = frame[(frame["set"] == "sink") & (frame.condition == "NONE")].set_index("question_id").cut_rule
    assert (frame.groupby("question_id").cut_rule.nunique() == 1).all()
    excluded = sink.index[~np.isfinite(sink.to_numpy()).all(1)].tolist()
    sink = sink.drop(index=excluded)
    sd, nd = sink.subtract(sink.NONE, axis=0), normal.subtract(normal.NONE, axis=0)
    assert list(sd.columns) == list(nd.columns)
    columns = list(sd.columns)
    sb = bootstrap_means(sd.to_numpy(), B, seed+100)
    nb = bootstrap_means(nd.to_numpy(), B, seed+101)
    results, all_controls = {}, {}
    for c, family in cfg["control_families"].items():
        controls = [x for x in columns if x.startswith(family + "_")]
        assert len(controls) == 10
        ic, ix = columns.index(c), [columns.index(x) for x in controls]
        matched_s, matched_n = sd[controls].mean(axis=1), nd[controls].mean(axis=1)
        bs = sb[:, ic] - sb[:, ix].mean(axis=1)
        bn = nb[:, ic] - nb[:, ix].mean(axis=1)
        pairwise = {}
        for control, j in zip(controls, ix):
            pairwise[control] = {
                "sink_candidate_minus_control": effect(sd[c]-sd[control], sb[:, ic]-sb[:, j], .99875),
                "normal_candidate_minus_control": effect(nd[c]-nd[control], nb[:, ic]-nb[:, j], .99875),
            }
        results[c] = dict(family=family,
            sink_delta_vs_NONE=effect(sd[c], sb[:, ic]),
            normal_delta_vs_NONE=effect(nd[c], nb[:, ic]),
            sink_minus_control_mean=effect(sd[c]-matched_s, bs),
            normal_minus_control_mean=effect(nd[c]-matched_n, bn),
            control_adjusted_sink_minus_normal=dict(mean=float((sd[c]-matched_s).mean() - (nd[c]-matched_n).mean()), ci=ci(bs-bn, .9875), ci_level=.9875),
            n_control_means_exceeded=int((sd[c].mean() > sd[controls].mean()).sum()),
            n_controls_exceeded_with_40way_CI=sum(v["sink_candidate_minus_control"]["ci"][0] > 0 for v in pairwise.values()),
            paired_controls=pairwise)
        for control, j in zip(controls, ix):
            all_controls[control] = dict(sink_delta_vs_NONE=effect(sd[control], sb[:, j]),
                                        normal_delta_vs_NONE=effect(nd[control], nb[:, j]))
    repeat_ids = [x for x in sink.index if rules[x] == "first_repeat"]
    repeated = {"n": len(repeat_ids), "ids": repeat_ids, "candidate_deltas": {}}
    if repeat_ids:
        sub = sd.loc[repeat_ids]
        for c, family in cfg["control_families"].items():
            controls = [x for x in columns if x.startswith(family + "_")]
            repeated["candidate_deltas"][c] = dict(mean_vs_NONE=float(sub[c].mean()),
                mean_minus_controls=float((sub[c]-sub[controls].mean(axis=1)).mean()),
                controls_exceeded=int((sub[c].mean() > sub[controls].mean()).sum()))
    return dict(sink_n=len(sink), normal_n=len(normal), excluded_sink_ids=excluded,
        cut_rule_counts_all_sink={str(k): int(v) for k,v in rules.value_counts().items()},
        candidates=results, controls=all_controls, detected_repeat_subset=repeated,
        units="mean per-token NLL, nats/token; positive delta lowers likelihood of fixed reference tokens")


def main():
    self_check()
    assert sha(CONFIG) == EXPECTED_CONFIG
    cfg = json.loads(CONFIG.read_text())
    source, out = ROOT / cfg["source_root"], ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    for filename, expected in cfg["source_sha256"].items():
        assert sha(source / filename) == expected
    B, seed = cfg["bootstrap_replicates"], cfg["seed"]
    a = behavior_analysis(pd.read_parquet(source / "step1_reencode.parquet"), cfg["behavior"], B, seed, out)
    b = likelihood_analysis(pd.read_parquet(source / "step2_screen_long.parquet"), cfg["likelihood"], B, seed)
    result = dict(analysis="post-hoc secondary, not a new confirmatory experiment", config_sha256=sha(CONFIG),
        code_sha256=sha(Path(__file__)), sources=cfg["source_sha256"], bootstrap_replicates=B,
        behavior=a, likelihood=b)
    write(out / "summary.json", result)
    write(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "summary.json"),
        pairs_sha256=sha(out / "behavior_pairs.json"), config_sha256=sha(CONFIG), code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(pooled=a["pooled"], movements={k:{arm:{kk:vv for kk,vv in v.items() if not kk.endswith('_ids')} for arm,v in value.items()} for k,value in a["movements"].items()},
        likelihood={k:{kk:vv for kk,vv in v.items() if kk != "paired_controls"} for k,v in b["candidates"].items()},
        cut_rules=b["cut_rule_counts_all_sink"], repeat_subset=b["detected_repeat_subset"]), indent=2))


if __name__ == "__main__":
    main()
