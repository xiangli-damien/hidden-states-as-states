# Sink C1：逐token能量匹配对照

**当前运行v3**：`sink-energy-matched-20260925-v3`。v2在307条后因单题总能量误差0.3398%超过原0.2%门槛停止，虽然逐token门槛已通过。v3将内部数值求解目标进一步收紧为每token 0.05% norm＋1e−7；原公开验收门槛和统计规则完全不变。v1/v2保留，未用于效应比较。当前配置／runner名称均以 `_v3` 结尾。

## BF16数值修订v2（新对照效应比较前）

首版在31/2352条记录后因一个token的BF16量化跳跃未达到1%norm容差而自动停止；首版失败与记录保存在原目录，没有计算新对照组间效应。v2使用新目录 `sink-energy-matched-20260925-v2`、新配置 `configs/sink_energy_matched_20260925_v2.json`、新脚本 `scripts/run_sink_energy_matched_v2.py`，原冻结文件保持不变。

不改变任何统计规则、方向、题目、容差。仅对无法通过标量二分匹配的token，沿既定方向的各坐标作确定性的相邻BF16舍入修正：在不超过目标能量的前提下，优先选择距同一理想方向的平方距离成本／增加能量最小的舍入。记录修正token数量、实际方向余弦；不声称BF16后的向量严格共线。此算法只看activation和norm，不看NLL。第二版仍必须通过原先所有能量门槛。

## 固定范围

Qwen2-7B-Instruct、原block14、原96道可评分sink B题与100道normal M_B题。沿用原始prompt、回答token、评分截断点、sink轴与75%分位阈值。执行NONE、C1与10个对照，共2,352次teacher-forced前向；不生成、不拟合、不重新分词或判分。模型和数据版本写入运行plan并校验哈希。

normal M_B的确按**原自动评分正确、且有完整boxed**选择；它不是独立人工验证的正确集。此前次要分析报告把它与早先自由生成实验“不要求正确”的normal组混写，现更正文档，NLL数值不变。旧分析冻结配置的文字元数据保持原文件与原SHA，并在本协议说明勘误。

## 控制变量

1. 原C1：`a_t=max(h_t·s−tau,0)`，`h'_t=BF16(h_t−a_t s)`。
2. 取原来的5个各向同性方向、5个非sink中心差方向，分别对s正交化、归一化；不重新抽样或按结果挑方向。
3. 各对照沿其正交方向删除分量，使用C1相同的实际活跃位置。采用确定性的标量二分搜索，使BF16实际改变量norm匹配同一token的C1实际norm。
4. 每token误差≤`1%×目标norm+1e−4`；每条回答总平方改变量相对误差≤0.2%；零改动位置必须完全相同。失败就停止，不丢题、不把不匹配结果当匹配结果。
5. 保存逐token目标norm、实际norm、对sink轴的实际余弦和logprob。方向在精确几何上正交，BF16实际改变量可能有少量偏离，单独汇报。

这是数值容差内的逐token匹配，不声称浮点运算中严格逐位等norm。二分仅访问activation和norm，不读取NLL或正确性来选幅度。

## 指标与判定

主指标沿用sink后缀平均NLL，正增量表示较少支持原参考续写；normal使用完整参考回答平均NLL。保留EOS、因果shift和原评分窗口。

同时满足以下条件才通过本轮预定方向检验：

- 全量实现／能量审计通过。
- C1减去10个对照均值的题目配对bootstrap98.75%区间下界>0。
- C1的点估计超过全部10个对照。

另报告全部10项配对99.5%区间（十项Bonferroni校正），明确区分“点估计超过每个”和“对每个均显著”。bootstrap100,000次，种子20260925。所有对照、正负结果和正常题影响都保存；不临时设副作用非劣效界限。

NONE与C1重新计算，并与原每题NLL比对，容差1e−5。初始实现检查使用A侧两道题，仅检查位置、恒等性、C1一致性和能量匹配。正式结果属于已看过数据上的新对照，不称为新样本确认。

无论检验结果如何，本实验都不能单独证明更高自由生成正确率、显著离开sink或恢复正常生成。

## 运行

当前配置：`configs/sink_energy_matched_20260925_v3.json`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /lambda/nfs/dami/openact/.venv/bin/python scripts/run_sink_energy_matched_v3.py --config configs/sink_energy_matched_20260925_v3.json
```

当前完整输出：`/lambda/nfs/dami/hss/sink-energy-matched-20260925-v3`。前两版原目录保留。先源码提交GitHub，Lambda fetch/ff-only后运行。现有MMLU已完成，本实验不抢占其他GPU任务。
