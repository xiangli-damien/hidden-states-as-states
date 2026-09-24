# GMM＋local PCA、固定分区FA与hard/soft MFA：有限范围的公平比较

## 冻结时间与范围

2026-09-23约17:33 UTC，在GSM8K新64题的功能确认结果尚未查看前固定本协议。当前已实现并启动**训练集上的固定分区FA拟合**；下面的联合MFA、功能评估是下一阶段，不能将拟合产物称作这些实验已完成。

配置：`configs/revision_fair_factors_20260923.json`。根目录：`/lambda/nfs/dami/hss/revision-fair-factors-20260923`。训练矩阵暂存SSD `/home/ubuntu/hss-revision-fair-factors-20260923`，拟合参数、来源与日志存dami。

## 数据和方法保持可比

- Qwen2-7B-Instruct，block14，前16个生成token的实际向量；原MATH训练问题、每题全部16位置。输入保持raw，float64用于数值计算，不做特征归一化。
- 第一批K=64固定，主rank=8，辅助rank=4/16。它检验固定预算的方法差异，不声称找到了全局最优K/rank。每个rank只有一个预定初始化，不能冒充多种子稳定性结论。
- 固定分区方法全部使用原GMM中心的最近中心编码器。原训练经验均值曾以float32保存；新PCA和FA统一使用由同一训练样本精确计算的float64经验均值，并记录与旧保存值的舍入差距。旧PCA结果和参数保留作桥接。
- 所有拟合只用训练问题；不以验证／测试KL、正确率或目标数据挑参数。

| 方法 | 分区／编码器 | 重构 | 每token连续编码量 |
|---|---|---|---|
| GMM＋local PCA | 固定最近GMM中心 | 共同经验均值＋正交低秩投影 | r，以及区域ID |
| 固定分区local FA | 同一固定分区 | 共同经验均值＋FA posterior mean重构 | r，以及区域ID；另报告噪声参数存储 |
| FA方向的正交投影，辅助 | 同一固定分区 | 投影到FA学到的W的列空间 | r，以及区域ID；隔离方向与posterior收缩 |
| Hard MFA | 联合MFA的最大posterior component | 该component的posterior mean | r，以及component ID |
| Soft MFA | 全部component posterior | 按responsibilities加权全部局部重构 | 原则上K×r坐标＋K−1权重，单列预算 |

不能把soft重构说成“同样8个坐标”。固定分区FA相对PCA增加了逐坐标噪声参数；hard MFA还改变了均值和分区。这些变化分别报告，不归咎于某一个因素。

## 训练和收敛

固定分区FA：每簇一个FA，初始化来自共同anchor下的PCA方向；逐簇对角噪声下限1e−5。使用已测试的精确float64 EM和有保护的SQUAREM，普通EM平均对数似然增量阈值1e−5，最多2000次EM map，每簇180秒预算。4个CPU worker，每个2线程；rank8先完成，再4/16。

每个拟合保存参数、全部似然轨迹、SQUAREM接受／拒绝次数、真实停止原因和收敛标记。额外计算一次普通EM更新验证stationarity；用独立numpy/Woodbury likelihood复核最终目标。若超时或未收敛，保留结果并明确标注，不能把预算停止当作收敛。合成数据上另用完整协方差高斯密度验证似然与共同anchor，CPU测试已通过。

联合MFA的预定下一阶段：同一训练矩阵、K64、r8/4/16，分别从对应固定FA参数初始化，使用component-specific diagonal noise（区别于其他MFA噪声约束）。同一有保护SQUAREM、tol1e−5、max2000 EM map，每个rank最多2小时GPU拟合；只有现有decoder释放GPU后才能开始。保存checkpoint和未收敛状态；不因目标功能效果追加初始化或挑更好结果。Hard/soft使用同一套联合拟合参数，仅解码方式不同。

## 功能评价：在看目标结果前固定

- 主FA比较：rank8固定分区FA − 新共同anchor local PCA的下一token KL及完整参考NLL。两项均报告，不能只择显著者。MFA、soft、FA正交投影与rank4/16是明确的次要比较。
- 原MATH局部研究选定的64题按32验证／32历史测试分别汇报；新GSM8K确认的64题使用同一固定解码器，不在目标上重拟合。目标题方法和预算在此预先确定，但不把后续改动方法继续称为同一次确认。
- block14、生成16后替换16位置；每个位置各自编码。保留未干预identity、原GMM center和原经验PCA桥接。
- 每条件fresh cache、once hook，精确核对真实patch与已保存capture；保留原始next-token logprobs、逐参考token NLL、bf16实际修改量、argmax与重新分配后的region。
- 问题级配对bootstrap，2000次，pointwise区间；验证／历史测试／新目标分开。不会因FA未收敛而丢弃落入这些簇的问题来美化功能指标；主结论要求方法收敛，否则明确称探索性未收敛结果。
- 分开报告训练／持出likelihood、MSE、输出KL/NLL、参数存储量、每token编码量和运行时间。更高likelihood不预设更低输出KL；参考保真不等于正确率或选择性steering。

固定分区、同anchor、同rank比较先回答因子模型是否值得替换简单PCA。后续K曲线、多个初始化、其他层和模型另立预算；不扩展为未经控制的全层网格搜索。
