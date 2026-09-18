"""Discovery/confirmation statistics for residual channels, not MLP neurons.

The unit of inference is one question, never an individual generated token.
All selection, scaling, and nuisance fits use the discovery partition only.
Cross-view FDR is applied after assembling the available planned views.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from hss.analysis.channel_data import ChannelData
from hss.experiments.artifacts import digest, file_digest, save_json, runtime_versions


def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1., posinf=1.)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1.)
    return out


def split_rows(rows, cfg):
    strata = rows.category.astype(str) + ":" + rows.y.astype(str)
    if strata.value_counts().min() < 2:
        raise ValueError("Too few observations for category/correctness stratification")
    train, test = train_test_split(np.arange(len(rows)), train_size=cfg["discovery_fraction"],
                                   stratify=strata, random_state=cfg["seed"])
    return np.sort(train), np.sort(test)


def welch(x, y):
    a, b = x[y == 1], x[y == 0]
    m1, m0 = a.mean(0), b.mean(0)
    v1, v0 = a.var(0, ddof=1), b.var(0, ddof=1)
    pooled = np.sqrt(((len(a)-1)*v1 + (len(b)-1)*v0)/(len(a)+len(b)-2))
    d = np.divide(m1-m0, pooled, out=np.zeros_like(m1), where=pooled>1e-12)
    se2 = v1/len(a) + v0/len(b)
    t = np.divide(m1-m0, np.sqrt(se2), out=np.zeros_like(m1), where=se2>1e-24)
    df = se2**2 / np.maximum((v1/len(a))**2/(len(a)-1)+(v0/len(b))**2/(len(b)-1), 1e-30)
    p = 2*stats.t.sf(np.abs(t), np.maximum(df, 1))
    return d, np.nan_to_num(p, nan=1.)


def nuisance(rows, train, x, *, retrospective=False):
    """Discovery vocabulary and moments only; includes current-vector log RMS."""
    columns = [np.ones(len(rows))]
    for field in ["category", "level", "first_token"]:
        values = rows[field].astype(str)
        counts = values.iloc[train].value_counts()
        # A common opening token can otherwise look like a correctness channel.
        minimum = 20 if field == "first_token" else 1
        known = counts[counts >= minimum].index.sort_values()
        mapped = values.where(values.isin(known), "__other__")
        levels = sorted(set(mapped.iloc[train]))
        for item in levels[1:]:
            columns.append(mapped.eq(item).to_numpy(float))
    numeric = [np.log1p(rows.n_prompt_tokens.to_numpy(float)),
               np.log(np.maximum(np.sqrt((x*x).mean(1)), 1e-12))]
    if retrospective:
        numeric += [np.log1p(rows.n_tokens.to_numpy(float)), rows.truncated.to_numpy(float)]
    for values in numeric:
        scale = values[train].std()
        columns.append((values - values[train].mean()) / max(scale, 1e-8))
    return np.column_stack(columns)


def residuals(x, y, c, train):
    coef_x = np.linalg.lstsq(c[train], x[train], rcond=1e-8)[0]
    coef_y = np.linalg.lstsq(c[train], y[train], rcond=1e-8)[0]
    return x-c@coef_x, y-c@coef_y


def correlation(x, y, df_adjustment=0):
    x, y = x - x.mean(0), y-y.mean()
    den = np.sqrt((x*x).sum(0)*(y*y).sum())
    r = np.divide(y@x, den, out=np.zeros(x.shape[1]), where=den>1e-12)
    df = max(len(y)-2-df_adjustment, 1)
    t = r*np.sqrt(df/np.maximum(1-r*r, 1e-15))
    return r, 2*stats.t.sf(np.abs(t), df)


def channel_stats(x, rows, train, test, cfg):
    x = np.asarray(x, dtype=np.float64)
    y = rows.y.to_numpy(int)
    d0, p0 = welch(x[train], y[train])
    d1, p1 = welch(x[test], y[test])
    rms = np.sqrt((x[train]**2).mean(0))
    rank = stats.rankdata(rms, method="average")/x.shape[1]
    peak_ratio = np.max(np.abs(x[train]), axis=0)/max(np.median(rms), 1e-12)
    lo, hi = cfg["middle_rms_quantiles"]
    middle = (rank >= lo) & (rank <= hi) & (peak_ratio < cfg["massive_ratio"])
    c = nuisance(rows, train, x)
    rx, ry = residuals(x, y, c, train)
    cr0, cp0 = correlation(rx[train], ry[train], c.shape[1])
    cr1, cp1 = correlation(rx[test], ry[test], c.shape[1])
    good = test[~rows.truncated.to_numpy()[test] & ~rows.parse_failed.to_numpy()[test]]
    dg, pg = welch(x[good], y[good])
    unit = x/np.maximum(np.sqrt((x*x).mean(1))[:, None], 1e-12)
    du, pu = welch(unit[test], y[test])
    return pd.DataFrame({
        "channel": np.arange(x.shape[1]), "rms": rms, "rms_percentile": rank,
        "peak_to_median_rms": peak_ratio, "middle": middle,
        "discovery_d": d0, "discovery_p": p0, "test_d": d1, "test_p": p1,
        "discovery_control_r": cr0, "discovery_control_p": cp0,
        "test_control_r": cr1, "test_control_p": cp1,
        "test_unit_d": du, "test_unit_p": pu,
        "test_clean_d": dg, "test_clean_p": pg,
        "n_train": len(train), "n_test": len(test), "n_clean_test": len(good),
    })


def views(cfg, name):
    data = ChannelData(cfg, name)
    last = data.info["model"]["n_layers"]-1
    for representation in ("prompt_last", "mean"):
        for layer in range(1, last+1):
            yield f"{representation}_L{layer}", data, representation, layer
        yield f"pre_{representation}", data, f"pre_{representation}", None
    token_marker = Path(cfg["cache_root"])/name/"tokens"/"_SUCCESS.json"
    if token_marker.exists():
        tokens = ChannelData(cfg, name, "tokens")
        if not np.array_equal(data.rows.sample_id, tokens.rows.sample_id):
            raise ValueError("Summary/token row identity mismatch")
        for norm in ("pre", "post"):
            for i, position in enumerate(tokens.info["positions"]):
                yield f"{norm}_t{position}", tokens, norm, i


def scaled_fit(x, y, indices, train):
    mu, sd = x[train][:,indices].mean(0), x[train][:,indices].std(0)
    sd = np.maximum(sd, 1e-8)
    z = (x[:,indices]-mu)/sd
    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(z[train], y[train])
    return model, mu, sd, model.decision_function(z)


def auc_ci(y, score, seed, repeats=500):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y==k) for k in (0,1)]
    vals = []
    for _ in range(repeats):
        take = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        vals.append(roc_auc_score(y[take], score[take]))
    return [float(x) for x in np.quantile(vals, [.025,.975])]


def probe_view(x, rows, table, train, test, cfg):
    """Fixed C/K, no test tuning. Channel rankings use discovery effects only."""
    y = rows.y.to_numpy(int)
    ranking = table[table.middle].sort_values("discovery_control_p").channel.to_numpy(int)
    out, predictions = [], pd.DataFrame({"sample_id": rows.sample_id.iloc[test], "y": y[test]})
    c = nuisance(rows, train, x)
    baseline = LogisticRegression(C=0.1, max_iter=1000).fit(c[train], y[train])
    score = baseline.decision_function(c[test])
    out.append({"kind": "nuisance_only", "k": 0, "auc": roc_auc_score(y[test],score),
                "ci": auc_ci(y[test],score,cfg["seed"])})
    predictions["nuisance"] = score
    for k in [1,4,16,64]:
        indices = ranking[:k]
        model, mu, sd, all_score = scaled_fit(x, y, indices, train)
        score = all_score[test]
        z = (x[:,indices]-mu)/sd
        aug = np.column_stack([c,z])
        augmented = LogisticRegression(C=0.1, max_iter=1000).fit(aug[train],y[train])
        predictions[f"k{k}"] = score
        out.append({"kind": "middle_channels", "k": k, "indices": indices.tolist(),
                    "auc": roc_auc_score(y[test],score), "ci": auc_ci(y[test],score,cfg["seed"]),
                    "augmented_auc": roc_auc_score(y[test],augmented.decision_function(aug[test])),
                    "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                    "train_mean": mu.tolist(), "train_std": sd.tolist()})
    return out, predictions


def temporal(cfg, name, tables, train, test, output):
    """Same first-token-selected channels, same long-response cohort at all times."""
    data = ChannelData(cfg,name,"tokens")
    rows = data.rows
    y = rows.y.to_numpy(int)
    selected = tables[tables.view.eq("pre_t1") & tables.middle].sort_values("discovery_control_p").head(16).channel.to_numpy(int)
    cohort = test[rows.n_tokens.to_numpy()[test] >= max(cfg["token_positions"])]
    first = data.array("pre",0)
    first = first[:,selected]
    records = []
    for i, position in enumerate(data.info["positions"]):
        x = data.array("pre",i)[:,selected]
        d, _ = welch(x[cohort],y[cohort])
        for j, channel in enumerate(selected):
            r = np.corrcoef(first[cohort,j],x[cohort,j])[0,1]
            records.append({"position":str(position),"channel":int(channel),"test_d":d[j],
                            "correlation_with_t1":r,"n":len(cohort)})
    pd.DataFrame(records).to_csv(output/"temporal.csv",index=False)


def category_checks(x, rows, table, train, test, cfg, output, view):
    """Unseen MATH categories and same-opening-token sensitivity."""
    y = rows.y.to_numpy(int)
    selected = table[table.middle].sort_values("discovery_control_p").head(16).channel.to_numpy(int)
    records = []
    for category in sorted(rows.category.unique()):
        take = test[rows.category.to_numpy()[test] == category]
        d,p = welch(x[take],y[take])
        for channel in selected:
            records.append({"category":category,"channel":int(channel),"d":float(d[channel]),
                            "p":float(p[channel]),"n":len(take)})
    pd.DataFrame(records).to_csv(output/f"categories_{view}.csv",index=False)
    dominant = int(rows.first_token.iloc[train].value_counts().index[0])
    take = test[rows.first_token.to_numpy()[test] == dominant]
    model,mu,sd,score = scaled_fit(x,y,selected,train)
    sensitivities = []
    for kind,indices in [("same_opening_token",take),
                         ("not_truncated_and_parsed",test[~rows.truncated.to_numpy()[test] & ~rows.parse_failed.to_numpy()[test]])]:
        if len(np.unique(y[indices])) < 2:
            continue
        sensitivities.append({"kind":kind,"n":len(indices),"first_token":dominant,
                              "auc":roc_auc_score(y[indices],score[indices]),
                              "ci":auc_ci(y[indices],score[indices],cfg["seed"])})
    # Length/truncation are downstream of generation: sensitivity, not causal adjustment.
    c = nuisance(rows,train,x,retrospective=True)
    rx,ry = residuals(x[:,selected],y,c,train)
    r,p = correlation(rx[test],ry[test],c.shape[1])
    sensitivities.append({"kind":"retrospective_length_control","indices":selected.tolist(),
                          "partial_r":r.tolist(),"p":p.tolist(),"q_selected":bh(p).tolist()})
    save_json(output/f"sensitivity_{view}.json",sensitivities)
    if len(rows.category.unique()) > 10:
        return
    logo = []
    for category in sorted(rows.category.unique()):
        tr = train[rows.category.to_numpy()[train] != category]
        te = test[rows.category.to_numpy()[test] == category]
        c = nuisance(rows,tr,x)
        rx,ry = residuals(x,y,c,tr)
        _,p = correlation(rx[tr],ry[tr],c.shape[1])
        rms = np.sqrt((x[tr]**2).mean(0))
        ranks = stats.rankdata(rms)/x.shape[1]
        lo,hi = cfg["middle_rms_quantiles"]
        candidates = np.flatnonzero((ranks>=lo)&(ranks<=hi)&(np.abs(x[tr]).max(0)<cfg["massive_ratio"]*np.median(rms)))
        ids = candidates[np.argsort(p[candidates])[:16]]
        model,mu,sd,score = scaled_fit(x,y,ids,tr)
        logo.append({"category":category,"n_train":len(tr),"n_test":len(te),"indices":ids.tolist(),
                     "auc":roc_auc_score(y[te],score[te]),"ci":auc_ci(y[te],score[te],cfg["seed"])})
    save_json(output/f"leave_category_out_{view}.json",logo)


def layer_updates(cfg,name,table,train,test,output):
    """Associational residual differences, not attention/MLP causal attribution."""
    data = ChannelData(cfg,name)
    y = data.rows.y.to_numpy(int)
    ids = table[table.view.eq("pre_prompt_last") & table.middle].sort_values("discovery_control_p").head(16).channel.to_numpy(int)
    last = data.info["model"]["n_layers"]-1
    previous = data.array("prompt_last",0)[:,ids].astype(float)
    records = []
    for layer in range(1,last+1):
        current = (data.array("pre_prompt_last") if layer==last else data.array("prompt_last",layer))[:,ids].astype(float)
        update = current-previous
        for j,channel in enumerate(ids):
            a,b = update[test[y[test]==1],j],update[test[y[test]==0],j]
            delta = a.mean()-b.mean()
            se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
            records.append({"layer":layer,"channel":int(channel),"correct_minus_wrong_update":delta,
                            "ci_low":delta-1.96*se,"ci_high":delta+1.96*se,
                            "state_mean_difference":current[test[y[test]==1],j].mean()-current[test[y[test]==0],j].mean()})
        previous = current
    pd.DataFrame(records).to_csv(output/"prompt_layer_updates.csv",index=False)


def transfer(cfg, name, table, train, test, output):
    """Source-only choice/fit; evaluate unchanged on held-out target questions."""
    if name != "llama32_math":
        return
    target_name = "llama32_mmlu"
    target_summary = ChannelData(cfg,target_name)
    _, target_test = split_rows(target_summary.rows,cfg)
    records = []
    for view in ["pre_prompt_last", "pre_t1", "pre_mean"]:
        if view not in set(table.view):
            continue
        phase = "tokens" if view == "pre_t1" else "summary"
        if not (Path(cfg["cache_root"])/target_name/phase/"_SUCCESS.json").exists():
            continue
        source, target = ChannelData(cfg,name,phase), ChannelData(cfg,target_name,phase)
        if source.info["model"] != target.info["model"]:
            # Model metadata may include irrelevant fields; basis/revision must match.
            for key in ["identifier","revision","hidden_dim","n_layers"]:
                if source.info["model"][key] != target.info["model"][key]:
                    raise ValueError("Incompatible cross-dataset channel basis")
        key, index = ("pre",0) if phase == "tokens" else (view,None)
        xs, xt = source.array(key,index).astype(float), target.array(key,index).astype(float)
        ys, yt = source.rows.y.to_numpy(int), target.rows.y.to_numpy(int)
        subset = table[table.view.eq(view) & table.middle].sort_values("discovery_control_p")
        ids = subset.head(16).channel.to_numpy(int)
        for mode in ["raw", "unit_rms"]:
            a,b = xs,xt
            if mode == "unit_rms":
                a = a/np.sqrt((a*a).mean(1))[:,None]
                b = b/np.sqrt((b*b).mean(1))[:,None]
            model,mu,sd,score = scaled_fit(a,ys,ids,train)
            ts = model.decision_function((b[:,ids]-mu)/sd)[target_test]
            records.append({"view":view,"mode":mode,"indices":ids.tolist(),
                            "source_test_auc":roc_auc_score(ys[test],score[test]),
                            "target_test_auc":roc_auc_score(yt[target_test],ts),
                            "target_ci":auc_ci(yt[target_test],ts,cfg["seed"]),
                            "n_target":len(target_test)})
        # Per-channel target effects retain the source-discovery sign.
        dt, pt = welch(xt[target_test],yt[target_test])
        per = subset.copy()
        per["target_d"] = dt[per.channel]
        per["target_p"] = pt[per.channel]
        per["target_q"] = bh(per.target_p)
        per["same_sign"] = per.discovery_d*per.target_d > 0
        per.to_csv(output/f"transfer_channels_{view}.csv",index=False)
    save_json(output/"transfer.json",records)


def run(cfg):
    root = Path(cfg["output_root"])
    root.mkdir(parents=True,exist_ok=True)
    version = file_digest(Path(__file__))
    all_summaries = []
    for ds in cfg["datasets"]:
        name = ds["name"]
        if not (Path(cfg["cache_root"])/name/"summary"/"_SUCCESS.json").exists():
            continue
        data = ChannelData(cfg,name)
        rows = data.rows
        train,test = split_rows(rows,cfg)
        destination = root/name
        destination.mkdir(exist_ok=True)
        split = rows.copy()
        split["partition"] = "confirmation"
        split.loc[train,"partition"] = "discovery"
        split.to_parquet(destination/"samples.parquet",index=False)
        frames = []
        for view,source,key,index in views(cfg,name):
            cache = destination/(view+".parquet")
            fingerprint = digest({"source":source.info,"analysis":version,"config":cfg,"view":view})
            marker = destination/(view+".json")
            if marker.exists() and json.loads(marker.read_text()).get("key") == fingerprint:
                frame = pd.read_parquet(cache)
            else:
                x = source.array(key,index)
                valid = np.isfinite(x).all(1)
                used_rows = rows[valid].reset_index(drop=True)
                # Retain the originally assigned split when late positions are absent.
                mask = np.zeros(len(rows),bool); mask[train]=True
                tr,te = np.flatnonzero(mask[valid]),np.flatnonzero(~mask[valid])
                frame = channel_stats(x[valid],used_rows,tr,te,cfg)
                frame["view"] = view
                frame.to_parquet(cache,index=False)
                save_json(marker,{"key":fingerprint})
                print(json.dumps({"analysis":name,"view":view,"n":int(valid.sum())}),flush=True)
            frames.append(frame)
        table = pd.concat(frames,ignore_index=True)
        for prefix in ["discovery","test","discovery_control","test_control","test_unit","test_clean"]:
            table[prefix+"_q_global"] = bh(table[prefix+"_p"])
        table["replicated"] = (table.discovery_q_global<=cfg["fdr"]) & (table.test_q_global<=cfg["fdr"]) & (table.discovery_d*table.test_d>0) & (table.discovery_d.abs()>=cfg["minimum_effect"]) & (table.test_d.abs()>=cfg["minimum_effect"])
        table["controlled"] = table.replicated & (table.discovery_control_q_global<=cfg["fdr"]) & (table.test_control_q_global<=cfg["fdr"]) & (table.discovery_control_r*table.test_control_r>0)
        table.to_parquet(destination/"channels.parquet",index=False)
        table[table.middle & table.replicated].to_csv(destination/"replicated_middle_channels.csv",index=False)
        summary = []
        for view,part in table.groupby("view",sort=False):
            middle = part[part.middle]
            summary.append({"view":view,"tested":len(part),"middle":len(middle),
                            "replicated_all":int(part.replicated.sum()),
                            "replicated_middle":int(middle.replicated.sum()),
                            "controlled_middle":int(middle.controlled.sum()),
                            "unit_middle":int((middle.replicated & (middle.test_unit_q_global<cfg["fdr"]) & (middle.test_unit_d*middle.discovery_d>0)).sum()),
                            "clean_middle":int((middle.replicated & (middle.test_clean_q_global<cfg["fdr"]) & (middle.test_clean_d*middle.discovery_d>0)).sum())})
        pd.DataFrame(summary).to_csv(destination/"summary.csv",index=False)
        probes = {}
        for view,source,key,index in views(cfg,name):
            if view not in {"pre_prompt_last","pre_mean","pre_t1","post_t1"}:
                continue
            x = source.array(key,index).astype(float)
            probes[view],predictions = probe_view(x,rows,table[table.view.eq(view)],train,test,cfg)
            predictions.to_parquet(destination/f"predictions_{view}.parquet",index=False)
            if view in {"pre_prompt_last","pre_t1"}:
                category_checks(x,rows,table[table.view.eq(view)],train,test,cfg,destination,view)
        save_json(destination/"probes.json",probes)
        if "pre_t1" in set(table.view):
            temporal(cfg,name,table,train,test,destination)
        transfer(cfg,name,table,train,test,destination)
        layer_updates(cfg,name,table,train,test,destination)
        status = {"name":name,"n":len(rows),"correct":int(rows.y.sum()),"train":len(train),"test":len(test),
                  "truncated":int(rows.truncated.sum()),"parse_failed":int(rows.parse_failed.sum()),
                  "views":len(frames),"model":data.info["model"],"split_seed":cfg["seed"]}
        save_json(destination/"status.json",status)
        all_summaries.append(status)
    save_json(root/"analysis.json",{"config":cfg,"datasets":all_summaries,"analysis_sha256":version,
                                   "versions":runtime_versions(),"scope":"Observational; no causal interventions."})
