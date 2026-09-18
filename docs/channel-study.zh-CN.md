# 正确性相关残差通道：研究协议与解释边界

## 研究对象

Llama-3.2-1B-Instruct × MATH（5,000），Qwen2-7B-Instruct × MATH（5,000），
Llama-3.2-1B-Instruct × MMLU（14,042）。仅消费发布、校验完成的 OpenAct 分片。
Qwen2 MMLU 仍在采集，不将部分 subject 当作全量结果。

这里的 index 是 hidden dimension 中的**残差流通道**，并非 MLP 中间层的 neuron index。
两个模型的坐标系不同；不能因为数字相同就认为是同一个特征。

- `prompt_last`：完整 chat template 输入的最后 token；其最后层状态用于预测首个回答 token。
- `t1`：首个回答 token 被送入模型**之后**的状态；可影响后续预测。
- `t2/4/8/16/32/64`：指定回答位置；不足长度的回答保留缺失值。
- `last`：最后生成 token，可能是 EOS，不等同于最后语义词。
- `mean`：完整回答 token 状态均值，包括特殊 token；属于回顾分析。
- 每层 `hidden_states` 的最后位置已经经过最终 RMSNorm；`final_norm/pre` 保留其输入。

## 预先固定的分析

配置 `configs/channel_study.toml`。固定种子 20260918，按题型/subject × 正误分层，
40% 发现、60% 验证。一个问题是一条独立观察，不能将数百万 token 当成独立样本。
阶段性输出的 FDR 会随完整计划视图加入而变化；完整报告才用于最终计数。

1. 激活规模仅在发现集定义。每个视图、每个通道计算跨题 RMS，20–95 百分位定义
   “中等幅度”；排除峰值超过通道 RMS 中位数 100 倍的位置。这是透明的操作性定义，
   **不是**文献对 massive activation 的充分判据，也不代表它们从未在其他位置出现极值。
2. 正确减错误的标准化均值差 Cohen's d，Welch 检验。发现、验证各自执行 BH 校正，
   每个模型数据集把全部计划层/时间视图/通道作为一个检验族。
3. “复现”：两边全局 q≤0.05、同号，且两边 |d|≥0.2。另报告全部幅度与中等幅度数量。
   这是通道计数，不是独立概念数量；跨层计数也不等于唯一 neuron 数量。
4. 混杂控制：题型、难度、prompt 长度、发现集常见首 token、当前向量 log RMS。
   仅在发现集拟合 nuisance 回归，对验证集残差计算相关和近似 t 检验、再全局 FDR。
   这是一种线性敏感性检查，不能保证已经控制全部题目难度/语义差异。
   首 token 用于 prompt 分析的控制时是生成后的变量，不能将该控制模型作为生成前预测器。
5. 另检验逐向量单位 RMS、未截断且成功解析的回答、相同首 token 的回答。
   回答长度/截断的进一步控制明确标记为回顾敏感性分析，不能解释为因果调整。
6. 在发现集按受控信号选 1/4/16/64 个中等幅度通道，固定 C=0.1 的 logistic probe，
   只在验证集报告 AUROC 和 500 次分层 bootstrap 的 95% 区间。选取 K 不使用测试成绩。
   同时报 nuisance-only 和加入通道后的结果；这些模型用于诊断，不声称部署时可用所有控制变量。
7. MATH 按每类留出：发现集中完全排除待测试题型，再选通道、拟合、测试该类的验证题。
8. 跨数据集：Llama MATH 的发现集选择与拟合完成后，冻结坐标、方向、缩放和权重，
   直接测试 MMLU 的验证集；raw 和 unit-RMS 分开呈现。MMLU 标签不参与源模型拟合。
9. 时间保持：冻结首 token 发现集选出的通道，使用同一批 ≥64 token 的验证回答，
   比较各位置的差异和与 t1 的跨题相关。不能把随时间变化的样本组成误认为保持/衰减。
10. 写入线索：在 prompt-last 上观察逐层 residual update = h[l] - h[l-1]；最后层使用
    pre-RMS。这只能定位“哪些 block 增量与差异相关”，不能区分 attention 与 MLP 因果贡献。

## 如何判断“统一存储”

同层、同一模型、同样通道、同样方向能够跨独立题型/数据集转移，是共享坐标的证据。
如果只有数据集内有效、跨数据集反向，不能称作统一正确性存储。
若很多坐标显著但低维 probe 就接近所有坐标的效果，它们可能共同编码少数方向。
相关通道数、预测所需通道数、独立维数、因果必要的通道数是四个不同量。

此外，自回归模型每个 token 都重新计算残差状态，过去 token 的信息主要通过 KV cache 被访问。
连续 token 上同一坐标有差异，并不能证明一个数值直接被复制/永久保存；可能是反复重新计算。

## 机制验证：尚未执行的下一阶段

现有采集没有 attention/MLP 输出或内部 neuron 激活。GPU 仍在全量采集，当前研究只用 CPU。
因此禁止把当前统计结果写成“神经元负责正确性”或“发现写入机制”。需另做：

1. 固定候选层/通道/方向；对一小组独立问题重复生成，取得**同一道题**的正确与错误轨迹，
   排除问题难度与内容变化。先检查标签及回答解析。
2. 在相同 teacher-forced token 上分解 attention 输出、MLP 输出；投影到候选残差方向，
   记录每个 block 的增量，进一步用 MLP down-projection 映射内部 neuron 对该方向的贡献。
3. 干预 prompt-last 或首生成位置：patch、消除该方向、正负微扰；与随机方向、幅度匹配通道、
   等范数扰动和 sham hook 对照。分开测 teacher-forced logit 变化与重新生成的正确率变化。
4. 对比只干预一次与每一步持续干预，检查后续信号恢复时间、KV 介导传播与再次写入。
   若信号恢复，继续定位提供补偿的 block/head；若答案改变，检查一般语言能力损坏。
5. 在另一题型和 MMLU 上冻结干预方案复现。无这些证据，只称“相关信号/可解码方向”。

## 参考

- Sun et al., [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)：
  巨大激活可能体现相对输入稳定的 bias 功能；大小和任务相关性应分别检查。
- Marks & Tegmark, [The Geometry of Truth](https://arxiv.org/abs/2310.06824)：
  跨数据集转移和因果干预是不同层次的证据，其事实真伪任务不等同于本研究的解题正确性。

## 运行

在 HSS 仓库中依次执行：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python scripts/study_channels.py --stage summary
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python scripts/study_channels.py --stage tokens
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python scripts/study_channels.py --stage analyse
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/study_channels.py --stage report
```

提取缓存位于 GPU 本地盘 `/home/ubuntu/hss-channel-cache`，表格、指标和报告位于
`/lambda/nfs/dami/hss/channel-study-20260918`。每个分片有源文件 hash、原始模型 revision、
采样位置和完成标记；统计缓存绑定源码 hash 与配置。无模型推理，无采集配置修改。
