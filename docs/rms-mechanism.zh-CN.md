# RMSNorm 后的“小范数”：可检验的机制与因果边界

## 1. 单个 token：范数反映方向经过 gamma 加权后的长度

令 x 是最后 block 的原始残差，d 是维数，Gamma=diag(gamma)。忽略很小的 epsilon：

```
u = sqrt(d) * x / ||x||
y = Gamma u
||y||² = d * sum_i gamma_i² * p_i
p_i = x_i² / sum_j x_j²
```

p_i 是该 token 在每个坐标上的能量比例。因此，输入的整体半径不决定输出范数；能量偏向小 |gamma_i| 的坐标时，post-RMS 的向量就短。gamma 对同一个模型的所有样本都固定，变化的是输入方向。不能说“模型为正确题选择了更小的 gamma”。

例：gamma=(0.2,2)，相同原始范数的两个向量分别指向第一、第二坐标，post-RMS 范数相差10倍。单纯把同一个 x 乘正标量，则输出不变（epsilon 和舍入误差除外）。这条数学机制并不依赖“正确性”。

也不能把“小 gamma”直接叫作“不重要”：预测还要乘 unembedding W。gamma 小的坐标仍可能通过较大的 W 权重影响 logits。类似地，post 隐藏态范数小不能直接推出输出熵大。

## 2. 完整响应：还包含 token 之间的抵消

```
||mean_t y_t|| = mean_t ||y_t|| * C
C = ||sum_t y_t|| / sum_t ||y_t||
```

C 在0到1之间。方向一致时接近1，抵消强时更小。它是带 token 幅度权重的一致性，不是纯粹的平均两两 cosine。

另一分解：

```
||mean_t y_t|| = ||mean_t u_t|| * G
G = ||Gamma mean_t u_t|| / ||mean_t u_t||
```

G 衡量 gamma 对响应均值方向的增益。两个分解分别回答：单 token 幅度还是跨 token 抵消；RMS-only 均值还是 gamma 的方向选择。对 log 范数取正确减错误的组均值，总差严格等于两个分量之和。它不是因果贡献率，分量可以互相抵消。

## 3. 三种候选解释

1. **方向重分配**：最后 block 让正确/错误回答在 gamma 大小不同的坐标上有不同能量。预测：gamma=1 时正误差异减小或反转，实际 gamma 恢复差异；打乱 gamma 的坐标对应关系会改变指标。
2. **响应内部方向抵消**：正确回答的 token 状态更分散或共同偏移更小。预测：单 token 的范数相近，但 C 更低；若主要来自长度，应在同长度/固定前16 token 上明显减弱。
3. **弱读出分量改变分母**：某些分量对 logits 的直接影响弱，却增加 RMS 分母，使其他输出分量相对缩小。这与 confidence-regulation 机制有关，但具体的 v 是否承担此功能必须实测。弱 unembedding 方向不等于小 gamma 方向。

参考：Stolfo 等《Confidence Regulation Neurons in Language Models》，https://arxiv.org/abs/2406.16254 。该文支持存在通过最终归一化调节 logits 的机制；不能据此把当前 Qwen 的差异直接归因于同一种机制。当前 v 来自 terminal-readout 工作分支的实际定义，在每个当前 checkpoint 上分别重算。

## 4. 本次执行的实验

- 三个已完成集合：Qwen2-MATH、Llama-3.2-MATH、Llama-3.2-MMLU（按原规则去重）。
- 只读取已验证 shard 的 final_norm/pre/per_token，CPU 重建理想 RMS，再与保存的真实 BF16 post 均值比较，逐题检查误差；无新生成。
- 四种固定响应算术条件：raw、gamma-only、RMS-only、RMS+gamma；另保留真实 BF16 post。中间层保持一致，重算 NDR/CoE。
- token 幅度与一致性分解；gamma 对均值方向的增益分解；同一批至少16-token 的样本比较全响应与固定前16 token。
- 20 次随机 gamma 坐标置换，保留权重分布；按 checkpoint 的 |gamma| 排序分成10组，对照方向能量分配。
- 测量已有 v 在每个 token 的能量比例，及仅从 RMS 分母平方和中移除 v 能量的算术对照。该对照保持分子固定，是路径敏感性，不是一次合法的完整 decoder 状态替换。
- 复用已冻结 40/60 分区，区间按问题 bootstrap 1000次。属于已见验证集上的探索性分析。

复现：依次运行 `scripts/study_rms_mechanism.py --stage extract`、`--stage analyse`；`--stage report` 只重新绘制报告。配置为 `configs/rms_mechanism.toml`。

## 5. 目前能识别什么因果关系

可以识别：对固定 x 和固定响应，更换 gamma 或 RMS 运算会怎样改变范数与几何评分。这是确定性读出变换的机制。

不能识别：小范数是否导致正确，或模型是否有一个主动维护的正确性寄存器。题目难度、内容、词汇、步骤和回答长度都可能共同影响隐状态方向与正确性。全响应均值还包含已生成回答的后果。

若几何分解显示稳定候选，下一层证据才是上游 block 干预：在生成前的确定位置改候选分量，设相同扰动范数的随机方向与内容保持对照，重新采样回答并评估正确率、熵和输出分布。当前并未运行该实验；持续 GPU 采集保持原调度。
