# 公平 FA/MFA 比较：执行与检查

截至 2026-09-23 22:34 UTC，**拟合、CPU几何、真实7B模型的2,432条件功能比较及两级审计全部完成。** 功能前向514.77秒，完整结果见[结果记录](revision-fair-factors-results-20260923.zh-CN.md)和[本地报告](http://127.0.0.1:8782/report/index.html)。本轮未发现FA/MFA优于同rank局部PCA的功能保真优势。

## 固定范围

遵循此前冻结的[公平比较协议](revision-fair-factors-plan-20260923.zh-CN.md)：Qwen2-7B、block14、前16个生成位置，K64，主rank8、辅助4/16，无特征归一化。

- MATH 64题：32验证、32历史测试；GSM8K此前固定的64题独立汇报。
- 每题19条件：identity、GMM中心、共同经验中心、旧PCA8桥接，以及三个rank下的PCA、FA后验重构、FA方向正交投影、hard MFA、soft MFA。
- 总计 **2,432条件**，样本单位仍是 **128道题**。
- 固定FA与PCA使用同一精确float64经验中心；FA拟合留下的<1e−7均值算术漂移单独保留并记录。PCA基来自此前冻结的确定性randomized SVD拟合，不声称是无限精度的最优子空间。
- Hard/soft共用同一个联合MFA模型。Soft rank8的编码预算为575个连续值，hard为8个加component ID；噪声与编码中心的存储量另计。

## 已实际通过的检查

1. 六项NumPy数值测试：完整协方差高斯密度与条件均值、FA后验收缩与正交投影的区别、潜变量旋转/分量置换不变性、零rank/退化方向、无效噪声拒绝、编码预算。
2. 两项统计测试：按问题配对、两个主指标均报告、重复问题不能作为独立样本。
3. OpenAct模型环境中的独立CPU检查：随机初始化小Qwen上的真实非恒等FA hook，带KV cache的逐段参考损失与完整因果前向逐token损失一致，最大误差0。它是实现检查，不是7B模型的实验结果。
4. 真实拟合参数与128题输入已经按SHA冻结，CPU计算全部2,432条件的理想/bf16几何，以及48,176训练token的完整混合似然。用独立既有MFA实现核对真实高维参数，密度最大误差3.64e−12；联合模型全训练似然与原拟合目标一致。
5. 所有FA列空间正交投影的理想欧氏误差均不大于同一FA后验重构误差，验证两种重构没有被混淆。

本地与远端8项pytest通过；本地/HSS环境缺少模型依赖的一项由OpenAct环境中的独立CPU脚本实际执行通过。没有升级环境。CPU几何/密度阶段311.3秒，输入准备10.7秒。

## 队列与结果

根目录：`/lambda/nfs/dami/hss/revision-fair-factors-20260923/comparison`。

- `plan.json`：输入、模型参数、条件、样本ID与SHA。
- `decoders.npz`、`fitting_summary.json`、`budgets.json`：可复用参数、收敛与预算。
- `inputs/{math,gsm8k}`：逐题原文、reference token IDs、真实capture。
- `geometry/`：已完成CPU结果与`audit.json`；不包含功能结论。
- `smoke/`、`functional/`：2题38条件冒烟和全量2,432条件均完成且审计通过。
- `report/`：PNG/PDF、两个主指标、全部辅助比较、预算/似然表与128道原题页面均完成；独立复算285条汇总、270条配对、6条主比较通过。

队列初始PID **445332**，脚本`run_revision_fair_comparison_queue.py`，当前`complete`，不可重复启动。它核对OpenAct v2两模型均完成后按顺序执行。每个条件都使用fresh cache、一次hook、capture精确比对，并保存实际bf16替换向量、原始logprobs和逐reference-token损失。实际bf16补丁均已独立重算。

检查`queue_status.json`与对应阶段日志；**不要重复启动队列或编辑它即将使用的冻结科学代码**。若冒烟失败，保留失败证据后检查；不能把部分结果包装成全量成功。

核心输入计划SHA：`880e6815579b8585c1c533878da6f7240f74c74c33003481d4bdb2709e4412cb`。
CPU几何结果收据SHA：`f8fa8e4109719afe9c0e247a9ae4e2d7d8724de78af091b0c2399ef9b8637a4d`。

## 环境分工

HSS `.venv`：拟合、CPU几何、数学/统计测试、报告；有torch/scipy/threadpoolctl，当前没有transformers。

OpenAct `.venv`：真实模型前向、原始执行审计、CPU小模型hook检查；有transformers，但没有threadpoolctl。队列通过进程环境变量限制BLAS线程，模型执行器不依赖threadpoolctl。

全部源代码经本地测试→GitHub→Lambda fetch/ff-only。GSM8K采集和本轮功能比较均已释放GPU。后续记号交换在能力门槛未通过后按协议停止，未运行test或真实干预。下一条独立应用线见同prompt多次采样协议，不重跑已完成拟合。
