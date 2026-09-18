# 公共分量与 NDR 的归约检验

## 数据与边界

Qwen2-7B-Instruct MATH 5,000、Llama-3.2-1B-Instruct MATH 5,000，
以及 Llama-3.2 MMLU 去除重复 prompt 后 13,937 题。
沿用原 discovery / confirmation 40/60 分区，身份与 prompt hash 一一核对。
当前 confirmation 已用于前期分析，本轮明确为探索性后续。
不生成新回答、不修改采集器，保留所有生成 token（含 EOS）。

Qwen 的 k=2570 是用户提出的既有假设。Llama 的参照坐标取 discovery
中逐 token raw 二阶矩最高的一维，不按正误挑选；与前轮最小 |gamma|
坐标 1159 的选择标准不同。报告完整坐标排名，避免把“1159 不重要”
偷换为“Llama 没有大激活坐标”。坐标从零开始编号。

## 必须区分的四个占比

令 x_t 为末层 RMSNorm 之前的 token 状态，bar x=mean_t x_t，
u_t=x_t/sqrt(mean_j x_tj²+eps)。

- q_raw_mean = bar x_k² / ||bar x||²。
- q_token_mean = mean_t(x_tk² / ||x_t||²)，主检验。
- q_energy_ratio = mean_t x_tk² / mean_t ||x_t||²。
- q_rms_mean = (mean_t u_tk)² / ||mean_t u_t||²，仅作与前轮衔接的诊断。

前三者直接使用 raw x，无 gamma、无 RMSNorm。归约不能通过混用这四项成立。
尤其在跨 token 的尺度或方向变化时，先平均和先算占比不交换。

## 幅度与“内容能量”

分别报告 |bar x_k|、mean_t |x_tk|、sqrt(mean_t x_tk²)；
以及去掉该坐标后的回答均值 norm 和逐 token 剩余平方能量。
剩余 raw 能量不是恒等的 1−q；只有归一化后的剩余**比例**才是 1−q。

统一中心化：mu = discovery 题目 bar x 的平均，每题等权、与标签无关。
E_centered = mean_t ||x_t−mu||² / D。
精确分解 E_centered = mean_t ||x_t−bar x||²/D + ||bar x−mu||²/D。
即回答内变化 + 回答均值偏离公共均值。两者都不自动具有“语义内容”含义。
first16 对照使用同一个冻结 mu；短于16 token 的回答缺失，不复制末 token。
固定位置使用 prompt_last、t1/2/4/8/16/32/64/last；不同位置可用的题数会变，
表中保留 n。prompt_last 是预测第一个生成 token 的状态，t1 是读入它之后的状态。

## 指标增益

NDR = mean_l ||bar h_l|| / ||bar h_L||。末层 norm 不是 NDR。
设 q_u = q_rms_mean，G_rest² 为除 k 外坐标 gamma² 的 u 均值能量加权平均：

||mean_t(gamma*u_t)||² = ||mean_t u_t||² [gamma_k² q_u + G_rest²(1−q_u)]。

因此单个比例相同不必得到相同 NDR：其余方向的 gamma 权重、跨 token 抵消、
以及 NDR 的其他层分子仍会变化。重构恒等式需在每道题上数值核对。

报告固定方向的 scalar AUROC：q/坐标幅度/NDR 越大预测正确，
能量或末层 norm 越小预测正确，不在验证集上翻转方向。
组均值和均值差 CI 也在原 60% 分区报告。

q_token、三种 raw q 联合、NDR，各自以及联合做 L2 logistic：固定三次 log/logit
特征，在 discovery 内四折选择 C，标准化只使用每个训练折。
对同一验证题做配对 bootstrap 比较增益。controls 包括题型、难度、prompt
长度、raw 回答均值 RMS、回答长度；后二者为生成后诊断，不是提前预测。
只检验此函数类下的补充预测能力，不证明全部信息等价。
预设描述性实际等价区间 ±.01 AUROC；区间跨零不表示落入该区间。
bootstrap 固定已拟合模型，不含重新训练不确定性。所有区间未做多重比较校正。

## 坐标排名和逐层定位

按 discovery raw RMS 排序，另存平均绝对值、每 token 为最大绝对坐标的频率，
以及每题等权 / 每 token 等权两种聚合。ties 的 argmax 取最低索引。
对 prompt_last、回答均值、t16 画全部层的绝对值与占比；slot0 为 embedding，
末 slot 替换为 raw pre-RMS，避免把最后一次 norm 当作 block 写入。
逐层观测不能区分 attention 和 MLP，更不能证明“确定时主动写入”。

当前没有全 prompt 各位置及 attention map，不能把大坐标直接命名为 attention sink。
[See What You Are Told, ICLR 2025，附录 A.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf)
列出 Qwen2-VL-7B 的 {458,2570}，仅作为相关模型的文献线索。

## 复现

在 HSS 仓库，配置为 configs/component_reduction.toml：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 15 .venv/bin/python scripts/study_component_reduction.py --stage extract
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 15 .venv/bin/python scripts/study_component_reduction.py --stage analyse
.venv/bin/python scripts/study_component_reduction.py --stage report
```

extract 可断点续跑，按来源和提取函数 hash 验证；只有完整缓存可以分析。
只解压最终层 pre-RMS token 状态，逐层图复用已经存在的 compact cache。
结果包含逐题标量、冻结预测、逐位置/逐层表、完整坐标排名、PNG/SVG 和来源版本。
