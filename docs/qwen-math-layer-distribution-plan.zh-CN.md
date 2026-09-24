# Qwen–MATH 全层向量分布诊断

## 问题与范围

研究为什么某些层在提高 ICL 百分比容差后仍保留较多 GMM 成分。
先把“分数零点带来的预算差异”与“点云本身的几何性质”分开，再判断是否值得进行更贵的重拟合对照。

本轮分析 Qwen2-7B-Instruct × MATH 全部5,000题的完整生成 token 均值，原始3,584维。
包含embedding0、block1–27，以及block28在最终RMSNorm前/后的两个独立表示，共30个位置。
聚类输入不做标准化、归一化、白化或PCA；计算协方差需要中心化，余弦需要方向归一化，
这些仅用于指标，不修改原始表示或现有模型。最终post均值来自逐token post状态的平均，
不是把RMSNorm施加到pre均值上。

## 可检验的解释

1. **计分效应**：各层ICL绝对值不同，让同样百分比对应不同的实际损失预算。
   重用所有K=2…80候选，报告 `ΔICL/(N*D)` 曲线和共同诊断阈值，原0–5%规则并列保留。
2. **公共方向与尺度**：全局均值占总能量比例、所有不同题对的平均余弦、norm分位数、
   最强坐标占总能量/中心化方差比例。区分公共偏置与实际题间变化。
3. **相关协方差**：全5000题、float64精确协方差和全谱分解；PR、entropy rank、d90/d95、
   前1/8/16方向方差比例、协方差Frobenius能量中的非对角占比。
   比较保留完整协方差与只保留对角项时的PR；后者仅是“去掉相关项”诊断，不是重新拟合。
4. **长尾与非高斯径向结构**：中心化半径分布、最外1%题的能量比例、坐标/PC峰度。
   用经验完整协方差的全部特征值生成三份单高斯的归一化平方半径，比较q99与第四矩。
   这是依赖已拟合协方差的参照，不报告p值，也不冒充给单高斯重新拟合GMM。
5. **簇分离与协方差表示能力**：复用GMM 0%、2%、5%和最终MFA rank16，
   检查样本内的簇间方差比例、簇大小、相同1024题的全3584维Euclidean silhouette。
   silhouette偏好某些簇形状，小值不能单独否定统计混合结构。
6. **长度与样本量敏感性**：精确计算仅移除log(1+回答长度)线性投影后的PR与解释方差比例。
   五次相同512题索引的无放回抽样比较PR；重复范围不是置信区间。

## 解释边界

- 全局PR和逐簇PR不是同一个对象，不能混画后当成同一个量。
- GMM成分数不是密度峰数、内在维数或已确认的功能状态数。
- 2D PCA仅作展示，拟合和silhouette仍在原始全维空间。
- 所有旧地图拟合过全部5000题；本轮属于描述性探索。
- 新分析不使用correctness标签。报告保留sample IDs和长度，以便后续定位原始问题。
- 本轮不新拟合GMM/MFA、不做跨层因果解释，也不宣称排除了所有长度/题型混杂。

## 审计与交付

每个缓存快照与原聚类快照严格匹配，样本ID和顺序一致；GMM/MFA须用同一个快照。
每个冻结分区重算64道题的posterior归属；全谱trace/Frobenius与直接协方差互相校验。
所有读取的状态、模型、分配及输入清单记录SHA256，结束时重验未变。
测试覆盖尺度/正交旋转不变量、方差分解恒等式、ICL共同阈值对分数整体平移的不变性。

配置：`configs/qwen_math_layer_distribution.json`。
分析：`scripts/study_layer_distribution.py --config configs/qwen_math_layer_distribution.json`。
报告：`scripts/report_layer_distribution.py --root /lambda/nfs/dami/hss/qwen-math-layer-distribution-20260924`。
CPU两线程；GPU上的干预实验继续。单层实测协方差约1秒、精确eigh约4.3秒；
加上径向对照、分区检查及保存，全层分析预计数分钟至十余分钟，不含代码和报告准备。

结果：`/lambda/nfs/dami/hss/qwen-math-layer-distribution-20260924`，同步本地results同名目录。
包含逐层表、逐分区表、ICL曲线、完整特征值谱、前32方向及投影、范数/半径、
单高斯径向参照、各分区标签、抽样索引、逐层图和可切换层的网页。

后续根据结果再设计等样本量、独立拟合的单高斯与正交旋转对照，以检验“成分主要补偿相关性”的解释。
当前这两个重拟合实验尚未包含在已执行范围内。

方法参照：[GMM协方差与模型选择](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)、
[PCA与方差解释](https://scikit-learn.org/stable/modules/decomposition.html)。
