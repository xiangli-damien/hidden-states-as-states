# Qwen MATH：原始激活的聚类与消融

## 范围与参数

- 来源：`Qwen/Qwen2-7B-Instruct` × MATH 全量 5,000 条；读取已验证的 OpenAct shards，固定 checkpoint revision 与样本顺序。
- 表示：每题所有生成 token 的 hidden mean。完整 3,584 维；embedding 加 28 个 decoder 输出。主图末层为模型原有的 post-RMSNorm，另做末层 pre-RMSNorm 对照。
- 不额外标准化、L2/RMS 归一化、whitening 或 PCA。`mfa_init="svd"` 只初始化每簇协方差因子，不变换或降维输入。
- 只运行 KMeans、对角 GMM、MFA。rank 为 `0,1,2,4,8,16,32`；rank=0 是 MFA 实现的对角高斯模型对照。
- GMM/MFA：3 次初始化，每次最多 2,000 次迭代，平均 log-likelihood 的绝对变化阈值 `1e-5`。MFA 优先保留已收敛初始化中似然最高者；没有收敛初始化时保留最好的未收敛结果作诊断。GMM 保留最高似然初始化。两种方法最终保留的拟合若未收敛，都不用于选 K。
- KMeans：10 次初始化、最多 1,000 次迭代，sklearn 相对中心移动阈值 `1e-4`。达到迭代上限的结果保守地标为未确认收敛。
- ICL 容差：`0%, 0.5%, 1%, 2%`。同时计算 BIC 对照。两种准则都越小越好，容差区间中选最小 K。KMeans 使用固定随机样本的 silhouette，不把它当作 ICL。
- 每个候选保存后验最大分配与最近中心分配。主结果采用模型本身的后验分配，最近中心分配作为论文式控制。KMeans 两者等价。
- 层间中心使用 cosine Hungarian、eta=0.6；这里 cosine 是匹配指标，不会归一化送进聚类的输入。

## 分阶段搜索

1. GMM、KMeans：所有层逐个扫描整数 K=2…80。
2. 固定每层 GMM 的 2% ICL 容差所选 K，做全部七个 MFA rank 对照。
3. MFA 每个 rank 独立扫描 K=`2,4,8,16,32,64,80`；再围绕每种 ICL/BIC 容差选择的 K，各向左右加密 2 个整数，做 2 轮。固定 K 阶段已有拟合复用。
4. 在搜索所得各策略选中的 K 上，用 seed=43、44 重拟合，保存两种分配的 ARI。

**MFA 是自适应加密搜索，不是 K=2…80 的全穷举，也不声称找到全局最优。** 每层已尝试的 K、未收敛/失败候选和实际选择都记录到 CSV。未收敛候选不定义最优 ICL，也不定义容差基准。

本研究使用全量回答做描述性聚类，正确性标签不参与选择。它不提供独立测试集上的正确性预测结论；改变 rank、K 或作图后不得将这些训练集描述称为泛化效果。

## 运行与恢复

代码按本地 commit → GitHub push → Lambda fetch/ff-only 的流程部署。Lambda 上运行：

```bash
cd /lambda/nfs/dami/hidden-states-as-states
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 10 .venv/bin/python -u scripts/run_cluster_ablation.py \
  --config configs/qwen_math_ablation.toml
```

配置分两份：`configs/qwen_math_raw.toml` 管数据和拟合；`configs/qwen_math_ablation.toml` 管 rank、选 K 策略和搜索阶段。主机使用最多 10 个 CPU worker，每个 2 个数值线程；总内存预算 96 GiB，保留至少 32 GiB 可用 RAM；不占采集 GPU。

同一命令恢复原任务。MFA 每 25 次 E-step 保存一次各初始化的原子 checkpoint，恢复时继续同一参数轨迹，不从头初始化。已完成的候选无需重拟合；改变 ICL 容差或后验/最近中心分配不重复拟合。协议、源数据或科学代码变化时必须新建 study，不能混入旧结果。

失败候选会记录错误并继续其他候选；它们不会被悄悄无限重试。若存在未收敛/失败拟合，最终状态为 `finished_with_unresolved_fits`，不是成功收敛。此时应检查原因，再明确决定迭代上限/正则化的新实验，而不是放宽阈值制造成功标志。

## 结果位置与读取

根目录：`/lambda/nfs/dami/hss/qwen-math-ablation-20260918`。

| 路径 | 内容 |
|---|---|
| `protocol.json` | 冻结参数、代码来源和搜索边界 |
| `study.json` | 当前阶段、活动任务、完成/收敛/错误计数 |
| `snapshots/*_rows.parquet` | 所有候选分配使用的样本顺序和题目元数据 |
| `candidates/*.json` | 每个 view/layer/method/rank/K/seed 的状态与模型路径 |
| `cache/fits/*/model.npz` | 所有候选的最佳初始化模型参数 |
| `cache/fits/*/assignments.npz` | `posterior` / `nearest`，均一题一个簇编号 |
| `cache/fits/*/fit.json` | likelihood、BIC、ICL、参数量、收敛状态、完整 MFA 似然历史 |
| `cache/fits/*/restart_*.npz` | 每次 MFA 初始化的可恢复参数和历史 |
| `candidate_metrics.csv` | 所有候选，含未收敛/失败结果 |
| `selection_ablation.csv` | 每层/方法/rank/准则/容差选择的 K |
| `rank_ablation_matched_k.csv` | 固定同一 K 的 rank 对照 |
| `seed_stability.csv` | 独立初始化种子对照的 ARI 和双方收敛状态 |
| `trials/*` | 标准 HSS Result：全层 states.npy、模型、对齐和逐题元数据 |
| `index.html`、`*.png` | 各阶段结束后更新的表格与消融图 |

所有有限候选的分配均保存，即使未被 ICL 选中，也可以重新使用参数和标签画图。只有所需全部层都有可选的收敛候选时才导出完整层间图结果；部分层可以直接从 candidates/ 和 assignments.npz 检查。
