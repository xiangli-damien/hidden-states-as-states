# MMLU 两模型对角 GMM：完整结果

**Qwen2-7B-Instruct 和 Llama-3.2-1B-Instruct 均已完成，并通过独立审核。** 队列在 2026-09-25 **09:39:37 UTC**（纽约 05:39:37）结束；从启动起共 **2 小时 6 分 31 秒**，包含准备、等待前序 steering、拟合与导出。检查时 GPU 无计算进程；没有启动额外实验。

| 项目 | Qwen2-7B-Instruct | Llama-3.2-1B-Instruct |
|---|---:|---:|
| 完整 MMLU 样本数 | 14,042 | 14,042 |
| post 存储层 | 0–28 | 0–16 |
| 另存的最终 pre-RMSNorm | 28 | 16 |
| 层／表示组数 | 30 | 18 |
| K 候选数 | 2,379 | 1,386 |
| 至少一次初始化收敛的候选 | 2,379 | 1,386 |
| 收敛初始化／总初始化 | 7,122 / 7,137 | 4,136 / 4,158 |
| 拟合＋导出用时 | 82.8 分钟 | 37.7 分钟 |
| 2% 容差的 post K 范围 | 7–56 | 3–46 |
| 2% 容差的末层 post K | 29 | 43 |
| 2% 容差的末层 pre K | 24 | 42 |
| 可加载 HSS 导出数 | 10 | 10 |

每个 K 均跑三次初始化。合计 37 次未收敛初始化全部留档，但不参与选择；每个候选都有其他收敛解。因此“所有 K 候选可用”不等于“每次初始化都收敛”。

## 1. 冻结的拟合口径

两个模型均使用完整回答的 **raw token mean**，只拟合对角 GMM，没有额外 normalization、标准化或 PCA。存储层 0 是 embedding；中间层是 block 输出；最终 post 层包含模型本身的 RMSNorm，另对最终 pre-RMSNorm 单独拟合。

ICL 越小越好，定义为 BIC + 2 × 后验熵。容差 t 选择满足 `ICL ≤ 最小 ICL + t × max(|最小 ICL|, 1)` 的最小 K。分别保存 **0%、2%、3%、4%、5%** 五种结果，2% 为主要展示版本。

K 先粗搜到 160，再围绕 ICL 前三名和五种容差选择做两轮整数细搜。本次两模型均没有触发扩展到 240，最终也没有上界警告。实际搜索集合逐层保存；结果是**已测试候选中的选择，并非整数范围的穷举全局最优**。

全部 14,042 条参与无标签地图拟合，正确性标签仅保存在元数据中。没有在这轮训练检测器、计算独立测试 AUROC 或做 steering；不能把这些聚类结果表述为独立测试泛化性能。

## 2. Llama 的逐层 K

| 存储层／表示 | 0% | 2% | 3% | 4% | 5% |
|---|---:|---:|---:|---:|---:|
| 0 | 50 | 3 | 2 | 2 | 1 |
| 1 | 56 | 7 | 4 | 2 | 2 |
| 2 | 67 | 9 | 6 | 4 | 3 |
| 3 | 74 | 10 | 7 | 5 | 3 |
| 4 | 94 | 18 | 11 | 7 | 5 |
| 5 | 80 | 16 | 10 | 6 | 5 |
| 6 | 83 | 14 | 8 | 6 | 5 |
| 7 | 72 | 12 | 8 | 7 | 4 |
| 8 | 57 | 12 | 8 | 6 | 5 |
| 9 | 77 | 13 | 8 | 6 | 5 |
| 10 | 72 | 20 | 13 | 8 | 6 |
| 11 | 84 | 24 | 17 | 11 | 7 |
| 12 | 93 | 32 | 24 | 16 | 12 |
| 13 | 96 | 35 | 25 | 19 | 14 |
| 14 | 91 | 46 | 35 | 28 | 22 |
| 15 | 93 | 42 | 32 | 27 | 22 |
| 16，post-RMSNorm | 98 | 43 | 35 | 29 | 24 |
| 16，pre-RMSNorm | 95 | 42 | 34 | 30 | 26 |

Qwen 的全部 30 行、五种容差见 [Qwen 完整结果](mmlu-diagonal-qwen-results-20260925.zh-CN.md)。两模型的主要 2% 序列为：

```text
Qwen post 0–28:
7,15,21,22,27,23,29,32,27,43,46,36,43,34,38,31,38,38,37,38,56,49,49,38,37,33,28,25,29
Qwen pre_final28: 24

Llama post 0–16:
3,7,9,10,18,16,14,12,12,13,20,24,32,35,46,42,43
Llama pre_final16: 42
```

Llama 的容差版本在后段层数较多，Qwen 的 L20 对 2–5% 容差不敏感。这些是拟合曲线的观察；K 同时受分布形状、对角协方差限制、ICL 惩罚和优化影响，不能直接当作真实计算状态数量。两模型的相同层编号也不代表相同相对深度。

## 3. 审核范围与保存位置

独立 CPU 审核验证了全部 3,765 个候选的标量记录、重启元数据、最佳收敛重启和 ICL 公式，重新得到全部 **240 个层／容差选择**。全部 **20 份 HSS 导出**通过完整文件哈希和样本／层／状态检查；对主要 2% 的全部 48 组数组进一步检查后验概率、MAP 簇 ID、导出一致性和恒等预处理。未重读非选中重启的全部大数组，也未重新拟合。

两张逐层 K 图都已实际查看。轻量文件已同步到本地，大 activation 和模型保留在 Dami。

- 远端根目录：`/lambda/nfs/dami/hss/mmlu-diagonal-gmm-20260925`
- 本地轻量结果：`/Users/lixiang/Projects/hidden-states-as-states/results/mmlu-diagonal-gmm-20260925`
- 各模型目录：`qwen2/`、`llama32/`
- 全部候选：`candidates/{view}/layer_XX/k_XXX/`，含三次初始化、最佳模型、簇 ID 和后验概率。
- 指标与选择：`candidate_metrics.csv`、`selection.csv/json`。
- 可加载地图：`exports/post_icl_0.02/`、`exports/pre_final_icl_0.02/`；其他容差分别存放。
- 审核：各模型的 `independent_audit.json`；图：`k_by_layer.png`。

读取已有模型无需重新拟合：

```python
from hss.results import Result, load_layer

root = "/lambda/nfs/dami/hss/mmlu-diagonal-gmm-20260925/llama32/exports/post_icl_0.02"
result = Result(root)
assert result.validate(full=True)["valid"]
model, projection, selection = load_layer(root, 14)
# X: Llama block14 原始完整回答 token mean，[N, 2048]。
cluster_ids = model.predict(projection.transform(X))
```

跨层 ID 仍用 cosine Hungarian、η=0.6 配对；这种编号对应不是已证实的跨层功能同一性。

完成凭据 SHA256：

- Qwen：`a7b70ffb35f97b655435e56c11d1796fc9ffba875de0124734777f9024bfc83a`
- Llama：`434532eb9a3616ae5262de263e7e7ca4c82897289d50a2d08941adff477c3bed`

本轮已授权工作完成，后续不自动扩大 K、增加模型或启动预测实验。
