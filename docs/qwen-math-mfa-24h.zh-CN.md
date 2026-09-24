# Qwen MATH：24 小时 MFA 交付

用户于 2026-09-20 03:15 UTC 要求将数周搜索改为 24 小时内可查看的结果。
截止时间固定为 **2026-09-21 03:15:13 UTC**（纽约 9 月 20 日 23:15）。

## 保留的数据与实验约束

- Qwen2-7B-Instruct，MATH 5000 题，原始生成回答 hidden mean，3584 维。
- embedding + 28 个层输出，额外一个末层 pre-RMSNorm 对照。
- 不额外 normalization、PCA、whitening；所有优化计算 float64。
- 历史候选保留原目录、参数、标签和来源。旧队列已在 107 个 MFA 候选处保存状态。
- 数据采集继续，单 GPU MFA worker；两个 CPU worker 并行准备初始化。

## 加速与验证

`src/hss/cluster/mfa_fast.py` 将输入常驻 GPU，按簇批量计算后验与充分统计量，
复用隐变量投影。普通 EM 数学更新与原实现一致；GPU 实测同一步数的参数相对误差
约 1e-13，末层短程提速约 2.6 倍。实际完整任务提速不等于短程提速。

另实现 SQUAREM 外推，参考 [Du & Varadhan, JSS 2020](https://www.jstatsoft.org/article/view/v092i07)。
权重与噪声必须有效，候选似然必须不低于两步普通 EM；否则缩短外推或回退。
停止条件始终检查普通 EM 的平均似然增量。
SQUAREM 会改变优化路径和局部最优，不声称与普通 EM 最终参数一致。
正式候选有三次初始化：前两次加速，最后一次普通批量 EM，保留已收敛结果中最高似然者。
它们是初始化敏感性控制，不是三个独立数据集重复。

记录：`/lambda/nfs/dami/hss/mfa-speed-20260919/batched_squarem*_20260920.json`。
末层 K21/r8 同一初始化：普通批量 EM 67.3s/1172步，加速 24.2s/303步，均达 1e-5；
中间层也更快，但加速后的似然较低，因而正式候选保留普通 EM 控制。

## 有限搜索

1. 保留已经完成的固定 GMM-K 对照拟合，作为额外候选和初步全层图。它们不能证明 MFA 的最优 K/rank；不再补跑固定 GMM-K 对照。
2. **每层的 rank4、8、16 全部独立搜索 K=2、8、16、32、64、80**。不根据固定 GMM-K 的结果淘汰任何 rank。完整粗网格为30个层/视图 × 3个rank × 6个K；已完成的同配置候选直接复用。
3. 每层、每个 rank 都在自己的当前最好 K 两侧做一轮局部加密，步长 max(1,K//5)，范围2–80。
4. 每层、每个 rank 最好的两个 K 做严格三初始化拟合。先覆盖所有层/rank 的第一名，再跑第二名。正式输出仅采用收敛、tol≤1e-5、三次初始化都完成的候选。各 rank 先得到自己的最佳 K，再比较 rank；不预先固定 rank 或继承 GMM 的 K。
5. 选中配置使用 seed1042、2042 做独立初始化敏感性检查，与原seed42初始化集合无重叠。

筛选：1初始化、最多400步、tol=1e-4，保留所有有限结果并显式标注。
正式：3初始化、最多2000步、tol=1e-5；筛选的较宽阈值不冒充正式收敛。
不收敛候选不会进入主 HSS；ICL/BIC 0%、0.5%、1%、2% 选择策略复用结果。
每层可采用不同 rank，同层各簇方向数量相同、具体方向独立。
这不是全网格或全局最优证明；主结果仅为全数据描述性聚类，不提供独立测试正确率结论。

时间预算保护最后的严格拟合和导出。每阶段记下全部计划、完成、未执行任务。
达到预算会停止新增任务并导出已有严格结果，状态明确区分完整搜索、预算截断、缺失层。
粗搜最迟在截止前10小时结束，加密最迟截止前8小时，严格拟合最迟截止前2小时。
阶段提前完成则立即进入下一阶段，不等待预算耗尽。最优仅指实际验证候选中的最优，不声称穷尽 K=2…80。

2026-09-20 用户指出固定 K 不应决定 rank，协议改为上述独立搜索。旧协议和阶段清单归档到
`protocol_history/`；完成拟合不变，原截止时间不变。首次应用改动须加 `--revise-search`，
只允许搜索策略变更，不允许用这个选项悄悄改变输入、拟合器或精度。

## 运行与读取

```bash
cd /lambda/nfs/dami/hidden-states-as-states
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 .venv/bin/python -u scripts/run_mfa_deadline.py \
 --source /lambda/nfs/dami/hss/qwen-math-ablation-gpu-20260919 \
 --directory /lambda/nfs/dami/hss/qwen-math-mfa-24h-20260920 \
 --deadline 2026-09-21T03:15:13Z --optimizer squarem
```

同一命令恢复；每个候选和每次初始化都有独立检查点。程序自身执行截止时间，
不依赖桌面监控在线才能停止搜索。

| 文件 | 用途 |
|---|---|
| `protocol.json` | 参数、截止时间、代码哈希、科学限制 |
| `study.json` | 当前阶段、活动拟合、全层覆盖、资源情况 |
| `candidates/*.json` | 每个候选的参数、来源、指标、模型位置 |
| `cache/fits/*` | 模型、逐题 posterior/nearest 硬标签、初始化检查点 |
| `selected_layers.csv` | 每层选定 K/rank；只有严格合格候选 |
| `rank_k_selection.csv` | 每层每个 rank 的独立最佳 K、已搜索 K、严格候选数与粗网格覆盖 |
| `selection_ablation.csv` | ICL/BIC 与容差敏感性 |
| `stability.csv` | 新种子的 ARI 和收敛状态 |
| `phases/*.json` | 搜索覆盖与预算截断明细 |
| `trials/*` | 标准 HSS Result，可独立 load、重新关联 ID |
| `index.html`、`figures.png/pdf` | 当前结果页面、参数曲线、末层搜索与稳定性图 |
| `delivery.json` | 最终交付状态和结果路径 |

`trials/*/config.json` 用 `cluster.rank_by_layer` 明确记录每层 rank。
每层保存实际 model/projection/selection；`states.npy` 为关联后状态，
`local_states.npy` 为未关联的逐层原始簇 ID，`rows.parquet` 固定样本顺序。
完成结果使用 HSS `Result.validate(full=True)` 校验，不需要重新拟合即可读取。

## 已完成结果与可重建报告

本轮于 **2026-09-20 11:19:51 UTC** 完成。粗网格540组（含3个缓存）、局部加密175组、
严格候选180组（含20个缓存）、独立种子检查60组全部执行。新增160个严格候选和60个
种子检查均收敛；2个筛选候选未收敛，未被冒充正式结果。

主策略为已验证严格候选中最低ICL、选择容差0。post共29个位置均选rank16，K范围4–9，
末层K7；pre-final对照rank16/K8。rank16是本轮搜索上限，不能据此断言它是全局最佳rank。
post种子ARI中位数0.609，末层0.488–0.495；pre-final为0.588–0.611。
这里检查的是固定配置下的初始化敏感性，不是重新选择K/rank的稳定性，也不是测试集预测。

`final_delivery_audit_20260920.json` 保存两套完整Result的校验、5000行样本ID/标签检查、
各层参数和种子统计。以下命令只读取已保存模型和表格，不读取原始激活、不重新拟合：

```bash
cd /lambda/nfs/dami/hidden-states-as-states
.venv/bin/python scripts/render_mfa_deadline.py \
  --directory /lambda/nfs/dami/hss/qwen-math-mfa-24h-20260920 \
  --samples /lambda/nfs/dami/hss/qwen-math-cluster-structure-20260919/exploration.json
```

报告写入独立的 `final_report/`，包含6组PNG/SVG图、逐层选择与稳定性表，以及按末层簇、
正确性和题号筛选的5000题原文/回答/状态轨迹。原文连接必须通过全部样本ID、正确性标签和
token数一致性检查。图表清单记录脚本、输入模型及原文文件的哈希；拟合后的trial不被改写。
本地可以在已同步的study目录执行同一命令，`--samples` 指向本地原文JSON。
