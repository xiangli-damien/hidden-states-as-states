# 长度、norm 和角度补偿

## 区分问题

当前距离报告的 length control 是对生成 token 数 T 的统计控制，另有向量 norm r。
没有重新归一化聚类输入。此实验将“角度”明确为生成均值 h 与所属簇保存中心 μ
在原始零点下的夹角 θ，s=||μ||。改变原点会改变角度，故不能把它当作无参考的语义方向。

精确恒等式：

```
||h-μ||² = (r-s)² + 2rs(1-cosθ)
```

- 排除 norm 尺度、保留方向：比较单位方向 chord `sqrt(2(1-cosθ))`，与夹角单调等价。
- 将角度贡献从距离平方中减掉：只剩 `(r-s)²`，是已有 norm 与中心 norm 的函数。
- 已知 r、θ 和簇编号（决定 s），距离被完全确定；预测器再加距离若有提升，
  只能反映有限模型表达交互的便利，而不是新的独立几何信息。

角度计算采用两个单位方向差的平方，再用 `2 asin(chord/2)` 得到 θ，避免小角度下
直接 `1-dot` 的相消误差。拒绝零向量的未定义夹角；原始 hidden 与原模型不修改。

## 复现

先运行距离分析，然后：

```bash
python scripts/analyze_cluster_angles.py \
  --report /lambda/nfs/dami/hss/qwen-math-cluster-structure-20260919 \
  --states /home/ubuntu/hss-cache/data/473f2199a8742bdf18996361/layer_28.npy \
  --rows /home/ubuntu/hss-cache/data/473f2199a8742bdf18996361/rows.parquet
python scripts/analyze_cluster_distances.py \
  --report /lambda/nfs/dami/hss/qwen-math-cluster-structure-20260919 --render-only
```

本地模型副本可用 `--fits-root .../inputs/fits`。模型哈希、矩阵哈希、行顺序及
旧欧氏距离均需一致。`angles.json` 保存各评分、OOF 预测、折号、精确公式误差、
代码与输入哈希；`angle_members.parquet` 保存逐题度量；`angles.html` 与
`angle_comparison.pdf/png` 用于查看。`--render-only` 不重新计算。

## 2026-09-19 观察

MFA rank8/K21 每题角度项 / 总距离平方的中位数为 96.39%；这不是正确性解释比例。
按簇内排名、固定“越小越正确”的单变量 AUROC：

| 度量 | AUROC |
|---|---:|
| 原始欧氏距离 | 0.5118 |
| 夹角或单位方向 chord | 0.4671 |
| 去除角度项后的径向差 | 0.5969 |

角度占据大部分几何变化，不代表携带最多的正确性信息。径向差的单变量改善，
需要与先前的 norm 效应区分，而非直接称为新信号。

固定几何下的 5 折 OOF 标签模型：MFA 的长度等组成控制 + norm 基线为 0.8234；
加入角度为 0.8240，增益约 0.0006，条件配对区间约 [-0.0004, 0.0016]。
加入去角度后的径向差为 0.8241，增益约 0.0007，区间约 [-0.0004, 0.0016]。
所以单变量的改善尚未转化为超出已有控制项的稳健增益。
再加欧氏距离相对 norm+角度的增益约 0.0005，区间跨零。
KMeans K2 的角度增益约 0.0055，而固定 K21 的三种方法角度增益区间都跨零，
因此结论依赖聚类粒度，不能概括为“角度对所有划分都无效”。

控制模型沿用距离报告：题型、难度、簇、生成长度、截断及 norm，固定样条与 C=1。
聚类本身已看过全部 5,000 条；bootstrap 固定 OOF 预测，没有重拟合聚类/预测器或
校正多项探索比较。所有结果是探索性描述，不是新数据上的可靠性或因果证据。
