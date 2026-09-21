# 跨层更新与跨 token 差值的 GMM

## 研究问题

当前研究先把状态聚类，再观察跨层路径。本实验直接聚类**表示的变化量**，检查变化是否存在可复现的分组，以及正确／错误回答在这些分组中的占比是否不同。主数据为 Qwen2-7B-Instruct × MATH 5,000。

两种不同的差值分别保存、分别建模：

1. **跨层更新**：`delta[l,t] = h[l,t] - h[l-1,t]`。对生成 token 平均后，每题每个 block 一个 3,584 维向量。线性恒等式保证其等于相邻层 hidden mean 之差，不需要重读数百 GB 的全部 token 激活。第 28 个 block 必须使用 `final_norm/pre`，因为 HF `hidden_states` 的末项已经经过最终 RMSNorm。
2. **跨 token 变化**：`delta[t] = h[t+1] - h[t]`。本轮先检查末层 pre/post 两种表示。每题均匀随机抽取八对不同的相邻生成 token，共 40,000 对，既不跨题目边界，也不混入 prompt→首生成 token 的边界。种子由 sample ID 决定。

这两种差值都不是“独立施加该向量就能复现行为”的因果作用。相邻 token 同时改变了位置、可见前缀和当前 token 身份。

## 表示与对照

| 表示 | 范围 | 单位 |
|---|---|---|
| mean_delta | 28 个 block | 一题的平均跨层更新 |
| prompt_delta | block 1/7/14/21/28 | prompt 最后一个 token 的跨层更新 |
| mean_state | 同五层 | 原始回答 hidden mean，作为静态基线 |
| prompt_state | 同五层 | prompt 最后位置状态，作为静态基线 |
| token_delta_pre/post | 末层 | 一对相邻生成 token 的差值 |
| token_mean_pre/post | 末层 | 全部相邻 token 差值的有符号均值 |
| gaussian_delta | block 7/14/28 | 单高斯模拟的平均更新量 |

共 47 个真实数据表示和 3 个单高斯对照。**token_mean 严格等于 `(last-first)/(T-1)`**，它只是端点基线，会受到回答长度分母的影响。不能把它当成中间变化过程的完整摘要。token_delta 的每题簇占比保留部分分布信息，但仍是八对样本的估计，不能冒充整段回答的完整状态轨迹。

## 拟合和选择

- 沿用固定 question-group 哈希划分：3,011 训练、1,003 验证、986 测试。同一题的所有 token 保持在同一集合。
- GMM 只在训练题目上拟合，输入原始差值，不 normalize、whiten、PCA 或按坐标标准化。
- 对角协方差；固定 `reg_covar=1e-5`；`tol=1e-3`；先 300 次 EM，未收敛者从参数继续最多 1,000 次；不收敛的候选保存但不能选中。
- 初始 K 为 1/2/4/8/16/32；若最优训练 ICL 触顶，则扩展 48/64/80。每 K 两次初始化。扩展决策是在第 14 层的**无标签试跑触顶之后、正误评价之前**确定，旧协议与试跑结果保留。
- 主选择为训练 `ICL = BIC + 2 * posterior_entropy` 最小；另报已搜索候选中的验证似然最优 K。粗网格不能声称精确最优 K，触顶结果明确标注。
- 稳定性：同 K 不同初始化，以及两次按整题随机取 80% 训练集重拟合；比较测试题目的 ARI。不会把“与自己 ARI=1”算成独立复现。
- 测试似然增益相对于 K=1 的对角高斯，以每维 nats 表示。区间按题目 bootstrap，不把同一题的 token 当作独立样本。
- temporal BIC/ICL 仍采用名义 token 对数作为样本数，未校正题内相关性；该 K 是探索性选择。验证和稳定性均保留整题分组。

[scikit-learn GMM 文档](https://scikit-learn.org/stable/modules/mixture.html) 说明了混合成分及协方差假设；[模型选择示例](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html) 展示了信息准则选择。GMM 分出多个成分本身不能证明有多个分离的密度峰。

## 单高斯对照的意义

设训练更新矩阵中心化后为 X。生成独立标准正态 G，并构造 `G @ X / sqrt(n_train) + mean`。模拟总体是一个高斯，其协方差恰为训练经验协方差 `X.T @ X/n_train`，保留相关方向，不需要 PCA。协方差可以是低秩的。

对角 GMM 可能用多个成分近似这片相关高斯。如果真实数据和该对照都分成多个成分，应优先把结论描述为“可由混合成分描述的几何异质性”，而不是已发现离散计算操作。每层只生成一份模拟，属于诊断，不是正式参数 bootstrap 的显著性检验。

## 正误评价

所有聚类和选择结束后再加载标签。主要指标为正确／错误题目平均簇占比的 JS divergence。跨 token 的每道题先独立归一化为直方图，再在题目间等权平均。

- 999 次测试题目标签置换。
- 条件对照一：按训练分位点定义回答长度四分位 × token entropy 四分位。
- 条件对照二：长度二分位 × entropy 二分位 × 题型 × 难度。
- 各检验家族分别对 47 个真实数据表示做 BH 校正。粗分箱不能完全排除混杂，不是因果检验。
- 跨 token 另报簇与 token 类型转换及生成相对位置的 NMI，检查分组是否明显反映数字、文字、符号、换行等。这里尚未充分控制具体词汇内容。
- 所有原文示例来自测试题目，在原始差值空间按簇内 Mahalanobis 距离选择；不按正误挑故事。

当前数据已多次被探索。这次新 GMM 不使用测试题目，并不使该数据重新成为完全未见的确认集。

## 执行与保存

配置：`configs/change-clusters.toml`。源代码按 local commit → GitHub push → Lambda fetch/fast-forward 同步。

```bash
.venv/bin/python scripts/run_change_clusters.py --stage prepare-summary
.venv/bin/python scripts/run_change_clusters.py --stage prepare-temporal
.venv/bin/python scripts/run_change_clusters.py --stage fit-depth
.venv/bin/python scripts/run_change_clusters.py --stage fit-temporal
.venv/bin/python scripts/evaluate_change_clusters.py
```

`prepare-temporal` 可以与 `fit-depth` 并行。CPU 工作进程和线程数独立配置。本轮不占 GPU，采集继续进行。

- SSD 特征缓存：`/home/ubuntu/hss-change-cache-20260921`。
- 结果：`/lambda/nfs/dami/hss/change-clusters-20260921`。
- `fits/<view>/k*_s*/` 保留每个候选的模型、均值、方差、权重、收敛信息。
- `fits/<view>/assignments.npz` 保存归属、后验、似然、原始题目索引。
- `selected_model.joblib` 可直接对同样定义的新差值调用 `predict` / `predict_proba`。
- `evaluation/` 保存正误统计、每题占比、簇描述、原文示例和 provenance。
- `report/` 保存可浏览 HTML 及 PNG/PDF。PCA 仅服务于二维展示，未参与 GMM 拟合。

尚未做：跨 token 全层扫描、全量 269 万 token 差值逐个标注、跨模型复现、不同抽样数量敏感性、旋转不变协方差族对照、正式单高斯 bootstrap、因果干预。
