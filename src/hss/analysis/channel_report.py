"""Scientific figures and a self-contained browsing page for channel evidence."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hss.analysis.channel_data import ChannelData
from hss.analysis.channel_study import split_rows, auc_ci
from sklearn.metrics import roc_auc_score
from hss.experiments.artifacts import save_json, file_digest


STYLE = {"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
         "axes.spines.right":False,"figure.dpi":120,"savefig.dpi":180,
         "axes.titleweight":"bold","axes.labelcolor":"#243342"}
COLORS = ["#77859a","#287e93","#d77543"]


def save(fig, root, name):
    for ext in ["png","svg"]:
        fig.savefig(root/(name+"."+ext),bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return name+".png"


def tag_image(dataset, filename, caption):
    return f'<figure><a href="{dataset}/{filename}"><img src="{dataset}/{filename}" loading="lazy"></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def table_html(frame):
    return '<div class="table">'+frame.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0,escape=True)+'</div>'


def example_page(cfg,name,data,x,ids,test,root,view):
    """Inspect extremes on the held-out partition; descriptive examples only."""
    ds=next(d for d in cfg["datasets"] if d["name"]==name)
    selections={int(j):np.concatenate([test[np.argsort(x[test,j])[:4]],test[np.argsort(x[test,j])[-4:]]]) for j in ids}
    wanted=set(data.rows.sample_id.iloc[np.unique(np.concatenate(list(selections.values())))])
    sources={}
    for shard in data.info["shards"]:
        frame=pd.read_parquet(Path(ds["path"])/shard/"data.parquet",columns=["sample_id","prompt_text","response_text"])
        for row in frame[frame.sample_id.isin(wanted)].to_dict("records"):
            sources[row["sample_id"]]=row
    chunks=[f'<h1>{html.escape(name)} · {view} 通道实例</h1><p>每个位置列出验证集激活最低和最高的各四题。这是理解内容的描述性切片；不能用它估计准确率。标签来自现有自动评估。</p><a href="../index.html#{name}">返回研究报告</a>']
    for j,indices in selections.items():
        chunks.append(f'<section><h2>Channel {j}</h2>')
        for i in indices:
            row=data.rows.iloc[i];src=sources[row.sample_id]
            title=f'{row.sample_id} | activation={x[i,j]:.4g} | {"正确" if row.y else "错误"} | {row.category} | {row.n_tokens} tokens'
            chunks.append(f'<details><summary>{html.escape(title)}</summary><h3>题目</h3><pre>{html.escape(src["prompt_text"])}</pre><h3>模型原始回答</h3><pre>{html.escape(src["response_text"])}</pre></details>')
        chunks.append('</section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>残差通道实例</title><style>body{font:16px/1.6 system-ui;background:#f4f6f8;color:#243342;max-width:1080px;margin:40px auto;padding:0 20px}section{background:white;padding:24px;border-radius:12px;margin:24px 0}details{border-top:1px solid #ddd;padding:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:18px;font:14px/1.7 ui-monospace,monospace}a{color:#17667d}</style>'+"\n".join(chunks)+'</html>'
    (root/"examples.html").write_text(page)


def dataset_figures(cfg,name):
    root = Path(cfg["output_root"])/name
    stats = pd.read_parquet(root/"channels.parquet")
    summary = pd.read_csv(root/"summary.csv").set_index("view")
    status = json.loads((root/"status.json").read_text())
    probes = json.loads((root/"probes.json").read_text())
    last = status["model"]["n_layers"]-1
    overview = ["pre_prompt_last",f"prompt_last_L{last}","pre_t1","post_t1","pre_mean",f"mean_L{last}"]
    labels = ["Prompt / pre-RMS","Prompt / post-RMS","Token 1 / pre-RMS","Token 1 / post-RMS","Mean / pre-RMS","Mean / post-RMS"]
    overview_labels = dict(zip(overview,labels))
    overview = [v for v in overview if v in summary.index]
    sensitivity=[]
    for view in overview:
        part=stats[stats.view.eq(view)]
        for low,high in [(.1,.99),(.2,.95),(.4,.9)]:
            keep=part.rms_percentile.between(low,high) & (part.peak_to_median_rms<cfg["massive_ratio"])
            sensitivity.append({"view":view,"low_rms_percentile":low,"high_rms_percentile":high,
                                "eligible":int(keep.sum()),"replicated":int((keep&part.replicated).sum()),
                                "controlled":int((keep&part.controlled).sum())})
    pd.DataFrame(sensitivity).to_csv(root/"amplitude_sensitivity.csv",index=False)
    figures=[]
    with plt.rc_context(STYLE):
        fig,ax = plt.subplots(figsize=(9,4.4),layout="constrained")
        y = np.arange(len(overview))
        ax.barh(y-.15,summary.loc[overview,"replicated_middle"],height=.28,color=COLORS[0],label="Replicated medium-amplitude channels")
        ax.barh(y+.15,summary.loc[overview,"controlled_middle"],height=.28,color=COLORS[1],label="Also passes nuisance controls")
        ax.set(yticks=y,yticklabels=[overview_labels[v] for v in overview],xlabel="Residual channels (not independent concepts)",title=name+" | held-out evidence")
        ax.invert_yaxis(); ax.legend(loc="lower right",fontsize=8)
        figures.append((save(fig,root,"counts"),"正误差异复现的通道数。中等幅度只用发现集定义；两侧 |d|≥0.2、同号且全局 FDR≤0.05。蓝色还通过题型、难度、首 token 等线性控制。"))

        fig,axs = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
        for ax,rep in zip(axs,["prompt_last","mean"]):
            seq = [f"{rep}_L{i}" for i in range(1,last)] + [f"pre_{rep}"]
            ax.plot(range(1,last+1),summary.loc[seq,"replicated_middle"],"o-",color=COLORS[0],label="Replicated")
            ax.plot(range(1,last+1),summary.loc[seq,"controlled_middle"],"o-",color=COLORS[1],label="After controls")
            ax.set(xlabel="Decoder block output (final = pre-RMS)",ylabel="Medium-amplitude channels",title=rep.replace("_"," "))
            ax.legend(fontsize=8);ax.grid(alpha=.15)
        figures.append((save(fig,root,"layers"),"逐层比较。最后一个 block 使用 pre-RMS，使层间差分不被最终归一化混淆。通道跨层重复，不应把这些数直接相加。"))

        view = "pre_t1" if "pre_t1" in set(stats.view) else "pre_prompt_last"
        part = stats[stats.view.eq(view)]
        fig,axs = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
        colors = np.where(part.middle & part.controlled,COLORS[1],np.where(part.middle & part.replicated,COLORS[2],"#cad1da"))
        axs[0].scatter(part.rms_percentile,part.test_d,c=colors,s=12,alpha=.7,rasterized=True)
        axs[0].axvspan(.2,.95,color=COLORS[1],alpha=.045)
        axs[0].axhline(0,color="#555",lw=.7)
        axs[0].set(xlabel="Channel RMS percentile (discovery only)",ylabel="Held-out correct - wrong (Cohen's d)",title=view+" | amplitude and correctness")
        axs[1].scatter(part.discovery_d,part.test_d,c=colors,s=12,alpha=.7,rasterized=True)
        lim=max(np.abs(part.discovery_d).max(),np.abs(part.test_d).max())*1.05
        axs[1].plot([-lim,lim],[-lim,lim],"--",color="#888",lw=1)
        axs[1].set(xlabel="Discovery d",ylabel="Confirmation d",title="Independent-split replication")
        figures.append((save(fig,root,"amplitude"),"灰色为其他通道，橙色为复现的中等幅度通道，蓝色为同时通过混杂控制。正值表示正确回答激活更大，负值相反。"))

        fig,ax = plt.subplots(figsize=(8.8,4.4),layout="constrained")
        for color,(key,entries) in zip([COLORS[1],COLORS[2],"#6755a1","#388459"],probes.items()):
            entries=[p for p in entries if p["kind"]=="middle_channels"]
            k=np.array([p["k"] for p in entries]); auc=np.array([p["auc"] for p in entries])
            ci=np.array([p["ci"] for p in entries])
            ax.errorbar(k,auc,yerr=[auc-ci[:,0],ci[:,1]-auc],marker="o",capsize=3,color=color,label=key)
        ax.axhline(.5,color="#aaa",ls="--");ax.set_xscale("log",base=4)
        ax.set(xticks=[1,4,16,64],xticklabels=[1,4,16,64],xlabel="Selected medium-amplitude channels",ylabel="Held-out AUROC",title="How compact is the readable signal?")
        ax.legend(fontsize=8);ax.grid(alpha=.15)
        figures.append((save(fig,root,"probe"),"通道选择、缩放与 probe 拟合只用发现集；误差线为验证集分层 bootstrap 95% 区间。四个 K 全部报告，没有用验证集选 K。"))

        increments=[(key,next(e for e in entries if e["k"]==16)) for key,entries in probes.items()]
        if all("delta_ci" in item for _,item in increments):
            fig,ax=plt.subplots(figsize=(8.8,3.8),layout="constrained")
            delta=np.array([item["delta_over_nuisance"] for _,item in increments])
            ci=np.array([item["delta_ci"] for _,item in increments])
            ax.errorbar(delta,np.arange(len(delta)),xerr=[np.maximum(0,delta-ci[:,0]),np.maximum(0,ci[:,1]-delta)],fmt="o",capsize=4,color=COLORS[1])
            ax.axvline(0,color="#888",ls="--");ax.set(yticks=range(len(delta)),yticklabels=[key for key,_ in increments],xlabel="AUROC gain: nuisance + 16 channels minus nuisance only",title="Is there additional information beyond the controls?")
            figures.append((save(fig,root,"incremental"),"与同一批验证题的 nuisance-only 模型做配对 bootstrap。区间跨 0 时，当前实验不能确认额外预测收益；基线包含题型/难度/首 token/向量 RMS 等，属于诊断比较。"))

        data=ChannelData(cfg,name,"tokens" if view=="pre_t1" else "summary")
        x=data.array("pre",0) if view=="pre_t1" else data.array("pre_prompt_last")
        train,test=split_rows(data.rows,cfg);y=data.rows.y.to_numpy(int)
        ids=part[part.middle].sort_values("discovery_control_p").head(4).channel.to_numpy(int)
        example_page(cfg,name,data,x,part[part.middle].sort_values("discovery_control_p").head(16).channel.to_numpy(int),test,root,view)
        fig,axs=plt.subplots(2,2,figsize=(10,6),layout="constrained")
        for ax,j in zip(axs.flat,ids):
            limits=np.quantile(x[test,j],[.005,.995]);bins=np.linspace(*limits,45)
            for label,color in [(0,COLORS[2]),(1,COLORS[1])]:
                values=x[test[y[test]==label],j]
                # Each class has unit total mass; tails outside the shown range are omitted.
                ax.hist(values,bins=bins,weights=np.full(len(values),1/len(values)),histtype="step",lw=1.6,color=color,label="Correct" if label else "Wrong")
            row=part.set_index("channel").loc[j]
            ax.set(title=f"Channel {j} | test d={row.test_d:.2f}",xlabel="Raw activation",ylabel="Fraction of class");ax.legend(fontsize=8)
        figures.append((save(fig,root,"distributions"),f"{view}：按发现集受控信号选出的前四个位置，展示独立验证分布。横轴显示合并分布 0.5–99.5% 区间；正误大量重叠时，不能逐题确定真假。"))

        temporal=root/"temporal.csv"
        if temporal.exists():
            frame=pd.read_csv(temporal,dtype={"position":str})
            positions=[str(t) for t in cfg["token_positions"]]+["last"]
            d=frame.pivot(index="channel",columns="position",values="test_d").reindex(columns=positions)
            r=frame.pivot(index="channel",columns="position",values="correlation_with_t1").reindex(columns=positions)
            fig,axs=plt.subplots(1,2,figsize=(11,6),layout="constrained")
            limit=max(float(np.abs(d.values).max()),.1)
            for ax,values,title,vmin,vmax,cmap in [(axs[0],d,"Correct - wrong (held-out d)",-limit,limit,"RdBu_r"),(axs[1],r,"Across-question correlation with t1",-1,1,"RdBu_r")]:
                im=ax.imshow(values,aspect="auto",vmin=vmin,vmax=vmax,cmap=cmap)
                ax.set(xticks=range(len(positions)),xticklabels=positions,yticks=range(len(values)),yticklabels=values.index,xlabel="Generated token position",ylabel="t1-selected channel",title=title)
                fig.colorbar(im,ax=ax,shrink=.8)
            figures.append((save(fig,root,"temporal"),f"始终用同一批 ≥64-token 的验证回答（n={frame.n.iloc[0]:,}）。保持同号不等于数值直接传播；last 可能为 EOS，与固定早期位置的语义不同。"))
            trained=next(p for p in probes["pre_t1"] if p["k"]==16)
            chosen=np.asarray(trained["indices"],int)
            token_data=ChannelData(cfg,name,"tokens")
            _,test=split_rows(token_data.rows,cfg)
            cohort=test[token_data.rows.n_tokens.to_numpy()[test]>=max(cfg["token_positions"])]
            yy=token_data.rows.y.to_numpy(int)[cohort]
            readout=[]
            for i,position in enumerate(token_data.info["positions"]):
                values=token_data.array("pre",i)[cohort][:,chosen].astype(float)
                z=(values-np.asarray(trained["train_mean"]))/np.asarray(trained["train_std"])
                score=z@np.asarray(trained["coef"])+trained["intercept"]
                ci=auc_ci(yy,score,cfg["seed"])
                readout.append({"position":str(position),"auc":roc_auc_score(yy,score),"ci_low":ci[0],"ci_high":ci[1],"n":len(cohort)})
            frozen=pd.DataFrame(readout);frozen.to_csv(root/"temporal_readout.csv",index=False)
            fig,ax=plt.subplots(figsize=(8.8,3.8),layout="constrained")
            ax.errorbar(range(len(frozen)),frozen.auc,yerr=[frozen.auc-frozen.ci_low,frozen.ci_high-frozen.auc],fmt="o-",color=COLORS[1],capsize=4)
            ax.axhline(.5,color="#888",ls="--");ax.set(xticks=range(len(frozen)),xticklabels=frozen.position,xlabel="Generated token position",ylabel="Held-out AUROC",title="Does a frozen first-token readout remain useful?")
            figures.append((save(fig,root,"temporal_readout"),"冻结 t1 发现集选出的 16 个通道、缩放和 probe 权重，在相同验证回答的后续位置直接读取；未重新训练或翻转方向。它检查同一读出方向是否持续可用，仍然不是因果传播实验。"))

        updates=pd.read_csv(root/"prompt_layer_updates.csv")
        mat=updates.pivot(index="channel",columns="layer",values="correct_minus_wrong_update")
        fig,ax=plt.subplots(figsize=(10,5.5),layout="constrained")
        limit=max(float(np.abs(mat.values).max()),1e-8)
        im=ax.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.set(xticks=range(0,last,2),xticklabels=list(mat.columns)[::2],yticks=range(len(mat)),yticklabels=mat.index,xlabel="Decoder block",ylabel="Prompt-selected channel",title="Where does the prompt residual difference change?")
        fig.colorbar(im,ax=ax,label="Mean block update: correct minus wrong",shrink=.8)
        figures.append((save(fig,root,"updates"),"逐层 residual 增量的组间差异。它只能定位相关 block；目前未采集 attention/MLP 内部输出，不能判断是哪一个模块写入，也不能证明因果作用。"))

        if (root/"transfer.json").exists():
            transfer=json.loads((root/"transfer.json").read_text())
            if transfer:
                fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
                yy=np.arange(len(transfer))
                auc=np.array([a["target_test_auc"] for a in transfer]);ci=np.array([a["target_ci"] for a in transfer])
                ax.errorbar(auc,yy,xerr=[auc-ci[:,0],ci[:,1]-auc],fmt="o",color=COLORS[1],capsize=3,label="MMLU (frozen MATH probe)")
                ax.scatter([a["source_test_auc"] for a in transfer],yy,color=COLORS[2],marker="s",label="MATH held-out")
                ax.axvline(.5,color="#aaa",ls="--");ax.set(yticks=yy,yticklabels=[a["view"]+" / "+a["mode"] for a in transfer],xlabel="AUROC",title="Do the same 16 channel coordinates transfer?")
                ax.invert_yaxis();ax.legend(loc="lower right",fontsize=8)
                figures.append((save(fig,root,"transfer"),"只在 Llama MATH 发现集选取和训练，冻结同一组通道、方向、缩放和权重后测试 MMLU。低于 0.5 表示原有方向在目标任务上反转；不做事后翻转。"))
    return figures,status,summary,stats,probes


def render(cfg):
    root=Path(cfg["output_root"])
    sections=[];manifest={};overall=[];headlines=[];persistence=[];increments=[]
    for ds in cfg["datasets"]:
        name=ds["name"]
        if not (root/name/"status.json").exists():
            continue
        figures,status,summary,stats,probes=dataset_figures(cfg,name)
        cards=f'<div class="cards"><div><b>{status["n"]:,}</b>独立 prompt（原始 {status.get("original_n",status["n"]):,} 条）</div><div><b>{status["correct"]/status["n"]:.1%}</b>正确率</div><div><b>{status["train"]:,} / {status["test"]:,}</b>发现 / 验证</div><div><b>{status["model"]["hidden_dim"]:,}</b>残差通道 / 层</div></div>'
        content=[f'<section id="{name}"><h2>{name}</h2>',cards]
        for filename,caption in figures:
            content.append(tag_image(name,filename,caption))
            manifest[f"{name}/{filename}"]=file_digest(root/name/filename)
        topview="pre_t1" if "pre_t1" in set(stats.view) else "pre_prompt_last"
        chosen=stats[stats.view.eq(topview)&stats.middle].sort_values("discovery_control_p").head(16)
        content.append('<h3>发现集选出的 16 个位置（0-based）</h3><p>按发现集受控信号排序，不按测试表现挑选。下表不是“已证实的正确性神经元”。</p>')
        content.append(table_html(chosen[["channel","rms_percentile","discovery_d","test_d","test_control_r","test_control_q_global","replicated","controlled"]]))
        sensitivity=pd.read_csv(root/name/"amplitude_sensitivity.csv")
        content.append('<h3>“中等幅度”的定义改变时，计数是否稳定？</h3><p>以下是同一组已校正统计的描述性阈值敏感性检查，主要定义仍为 20–95 百分位。</p>'+table_html(sensitivity[sensitivity.view.eq(topview)]))
        counts=summary.loc[[v for v in ["pre_prompt_last","pre_t1","post_t1","pre_mean"] if v in summary.index]].reset_index()
        counts.insert(0,"dataset",name);overall.append(counts)
        if "pre_t1" in probes:
            first=next(p for p in probes["pre_t1"] if p["k"]==16)
            base=next(p for p in probes["pre_t1"] if p["k"]==0)
            last_counts=summary.loc["pre_t1"]
            headlines.append({"数据组合":name,"独立问题":status["n"],"复现通道":int(last_counts.replicated_middle),
                              "受控后复现":int(last_counts.controlled_middle),"16通道 AUROC":round(first["auc"],3),
                              "AUROC 95% CI":f'{first["ci"][0]:.3f}–{first["ci"][1]:.3f}'})
            increments.append({"数据组合":name,"nuisance AUROC":base["auc"],"增加16通道 AUROC":first["augmented_auc"],
                               "增益 95% CI":f'{first["delta_ci"][0]:+.3f}–{first["delta_ci"][1]:+.3f}'})
        if (root/name/"temporal_readout.csv").exists():
            readout=pd.read_csv(root/name/"temporal_readout.csv",dtype={"position":str}).set_index("position")
            persistence.append({"数据组合":name,"t1 AUROC":readout.loc["1","auc"],"t2 AUROC":readout.loc["2","auc"],
                                "t16 AUROC":readout.loc["16","auc"],"t64 AUROC":readout.loc["64","auc"],"同一批验证题":int(readout.loc["1","n"])})
        sensitivity_path=root/name/"sensitivity_pre_t1.json"
        if sensitivity_path.exists():
            sensitivities=json.loads(sensitivity_path.read_text())
            content.append('<h3>首 token 检验的敏感性分析</h3>'+table_html(pd.DataFrame([{"subset":s["kind"],"n":s["n"],"AUROC":s["auc"],"95% CI":f'{s["ci"][0]:.3f}–{s["ci"][1]:.3f}'} for s in sensitivities if "auc" in s])))
        logo_path=root/name/"leave_category_out_pre_t1.json"
        if logo_path.exists():
            logo=json.loads(logo_path.read_text())
            content.append('<h3>完全未参与选通道/训练的 MATH 题型</h3>'+table_html(pd.DataFrame([{"category":s["category"],"test_n":s["n_test"],"AUROC":s["auc"],"95% CI":f'{s["ci"][0]:.3f}–{s["ci"][1]:.3f}'} for s in logo])))
        content.append(f'<p class="downloads"><a href="{name}/examples.html">逐通道查看原始题目与回答</a> · <a href="{name}/summary.csv">通道计数 CSV</a> · <a href="{name}/replicated_middle_channels.csv">全部复现通道 CSV</a> · <a href="{name}/probes.json">Probe 参数与置信区间</a> · <a href="{name}/channels.parquet">全部通道统计 Parquet</a> · <a href="{name}/samples.parquet">逐题 split 与元数据</a></p></section>')
        sections.append("\n".join(content))
    combined=pd.concat(overall,ignore_index=True)
    combined.to_csv(root/"overview.csv",index=False)
    findings='<section><h2>先看结论与证据</h2><h3>1. 首个生成 token 已存在正误相关差异</h3><p>以下只比较最后 block、最终 RMSNorm 之前的中等幅度通道。所有数字来自独立验证，多个通道不等于多个独立概念。</p>'+table_html(pd.DataFrame(headlines))
    findings+='<h3>2. 这是否超出了题型、难度等信息？</h3><p>相关信号不一定带来额外预测收益。下表比较同一批验证题，增益区间跨 0 时，不能确认有稳定的额外收益。</p>'+table_html(pd.DataFrame(increments))
    findings+='<h3>3. 同一组首 token 信号是否沿生成过程保持？</h3><p>冻结首 token 的选择、缩放和权重，不在后面重新寻找通道。对全部位置使用同一批足够长的回答；0.5 代表随机排序。</p>'+table_html(pd.DataFrame(persistence))
    transfer_path=root/"llama32_math"/"transfer.json"
    if transfer_path.exists():
        transfer=json.loads(transfer_path.read_text())
        frozen=[{"视图":t["view"],"MATH AUROC":t["source_test_auc"],"直接转移 MMLU AUROC":t["target_test_auc"],"MMLU 95% CI":f'{t["target_ci"][0]:.3f}–{t["target_ci"][1]:.3f}'} for t in transfer if t["mode"]=="raw"]
        findings+='<h3>4. 是否存在跨题集统一的坐标？</h3><p>下面冻结 Llama MATH 的 16 通道 probe，直接测试 MMLU。数据集内能读出与跨数据集能转移是两件事。另一模型 Qwen 的同号数字不代表相同神经元。</p>'+table_html(pd.DataFrame(frozen))
    findings+='<h3>5. 怎样写入和维持？当前证据还不够</h3><p>逐层图定位的是残差总更新与正误差异的关联。现有数据没有 attention/MLP 内部输出，也没有执行干预。下一步需要同题不同正确性轨迹、模块输出分解及带随机/等范数对照的 activation patching；方案见完整协议。</p></section>'
    save_json(root/"headline_findings.json",{"first_token":headlines,"incremental":increments,"persistence":persistence})
    cross=''
    if (root/"coordinate_overlap.json").exists():
        overlap=pd.DataFrame(json.loads((root/"coordinate_overlap.json").read_text())).drop(columns="same_direction_indices")
        cross='<section><h2>Llama MATH 与 MMLU：是否是同一批位置？</h2><p>两边各自的独立验证都成立，才计算交集；同时列出同号的数量。它与冻结 MATH probe 的跨任务转移是不同检验。</p>'+table_html(overlap)+'</section>'
    intro='''<header><div class="eyebrow">OPENACT / HSS · OBSERVATIONAL STUDY</div><h1>中等幅度残差通道中的正误相关信号</h1><p>两个模型、三个完整模型×数据集组合，独立发现与验证。先确认现象，再检验跨题型、跨数据集和时间保持。</p></header>
<div class="note"><strong>证据边界：</strong>本报告研究 residual channel，不是 MLP neuron。可读出正误相关信息不等于存储“真值”，也不证明该通道决定正确性。首生成 token 状态位于它被输入模型之后；prompt-last 才用于预测该 token。当前没有执行因果干预。</div>
<section><h2>检验标准</h2><p>先按完整 prompt 去重，再做 40% 发现 / 60% 验证。MATH 无重复；MMLU 14,042 条保留 13,937 个独立 prompt。中等幅度 = 发现集通道 RMS 的 20–95 百分位，并排除极端峰值。两边 |Cohen’s d|≥0.2、同号且所有层/位置的 BH-FDR≤0.05，才记为“复现”。“受控”另要求题型、难度、prompt 长度、首 token 和向量 RMS 控制后复现。<a href="protocol.md">完整协议与机制实验设计</a></p>
<p>这些是当前数据上的新研究，不是对原论文“真实性神经元”的复现。回顾均值含完整答案；不得作为生成前预测。计数中的多个坐标可能编码同一低维方向。</p></section>'''
    nav='<nav>'+"".join(f'<a href="#{html.escape(d["name"])}">{html.escape(d["name"])}</a>' for d in cfg["datasets"])+ '</nav>'
    css='''body{margin:0;background:#f4f6f8;color:#233140;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px}h1{font-size:38px;line-height:1.2;max-width:900px}h2{font-size:25px;border-bottom:1px solid #dce3e9;padding-bottom:12px}h3{margin-top:32px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#287e93}header p{font-size:18px;color:#556779}section{background:white;padding:28px;border:1px solid #e2e7ed;border-radius:14px;margin:26px 0;scroll-margin-top:60px}.note{padding:20px 24px;background:#e7f0f4;border-left:4px solid #287e93;border-radius:6px;margin:28px 0}nav{position:sticky;top:0;background:#f4f6f8ed;padding:14px 0;display:flex;gap:24px;z-index:5;backdrop-filter:blur(8px)}a{color:#17667d;text-decoration:none}a:hover{text-decoration:underline}.cards{display:flex;gap:25px;flex-wrap:wrap}.cards div{flex:1;background:#f6f8fa;padding:14px;border-radius:9px;min-width:120px;font-size:13px}.cards b{display:block;font-size:24px}figure{margin:28px 0 40px}figure img{width:100%;height:auto;border:1px solid #e4e8ed;border-radius:7px}figcaption{font-size:14px;color:#5d6c7b;padding:9px 2px}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:8px 10px;border-bottom:1px solid #e3e8ee}th{background:#f3f6f8;white-space:nowrap}.downloads{font-size:14px}footer{font-size:13px;color:#617180}'''
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>残差通道与正确性 · OpenAct / HSS</title><style>{css}</style><main>{intro}{nav}{findings}<section><details><summary><strong>完整结果概览：pre/post RMS 与 prompt/生成均值</strong></summary>{table_html(combined)}</details></section>{cross}'+"\n".join(sections)+'<footer>本报告保留全部计划视图、源数据身份、逐题划分、模型参数和统计表。代码与原始 collector 分离；原始 Zarr 未修改。<a href="analysis.json">分析配置 / 版本</a> · <a href="manifest.json">图表校验值</a></footer></main></html>'
    (root/"index.html").write_text(page)
    protocol=Path(__file__).resolve().parents[3]/"docs"/"channel-study.zh-CN.md"
    (root/"protocol.md").write_text(protocol.read_text())
    save_json(root/"manifest.json",manifest)
    save_json(root/"report_provenance.json",{"report_sha256":file_digest(Path(__file__)),
                "analysis_json_sha256":file_digest(root/"analysis.json"),
                "temporal_readout":"Freeze pre_t1 discovery-selected k=16 weights; same test cohort >=64 tokens; 500 stratified bootstrap draws."})
    print(str(root/"index.html"),flush=True)
