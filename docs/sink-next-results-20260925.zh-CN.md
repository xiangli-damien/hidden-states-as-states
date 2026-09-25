# HSS sink 推广实验：结果

更新：2026-09-25T16:50:05.174571+00:00。只展示完整、通过原始记录审核的实验。

[冻结协议](sink-next-protocol-20260925.zh-CN.md) · [此前 v3 能量匹配结果](sink-energy-matched-results-20260925.zh-CN.md)

**性质：探索性延伸，沿用此前已经检查过的问题；不是新样本确认实验。** NLL 单位为 nats/token，正差表示模型对既定参考续写的支持下降。

## 1. 同题型、同难度

96 对精确匹配；放宽一级 0 对；未匹配 0 条。两组都使用 half_sentence 后半段窗口。

| 量 | 均值 [95% CI] |
|---|---:|
| Sink：C1−NONE | 0.09165 [0.08063, 0.10263] |
| 匹配非 sink：C1−NONE | 0.00459 [0.00225, 0.00744] |
| 主终点：匹配对差 | **0.08706 [0.07638, 0.09768]** |

| 组 | 越界 token 比例（逐题均值） | 正超出量均值 | 实际更新范数均值 |
|---|---:|---:|---:|
| matched | 34.83% | 1.13246 | 1.13298 |
| sink | 66.56% | 4.58257 | 4.58309 |

结果超出已匹配的题型和难度。仍可能存在长度、回答内容等差异；两组自然截断的实际剂量不同，所以不把本项单独解释为完全排除题目因素或证明生成模式。

本项 sink NLL 与 v3 的数值不同，是因为按预定规格统一改为 half_sentence，而 v3 包含 first_repeat 窗口。

English: With problem type and difficulty matched in 96 independent pairs, the sink-minus-control difference in the clamp-induced NLL increase was 0.0871 nats/token (95% CI [0.0764, 0.0977]).

![固定人群与预选层/状态的效应](sink-next-20260925/effects.png)

## 4. 自由生成

已冻结、按依赖顺序运行；完整审核结果尚未同步，当前不报告部分效应。152道sink题+50道正常题，zero/C1/ORTH_MAN_1，共606条记录。

## 范围与追溯

第二个模型按用户规格留到讨论期。所有结果须在用户指定2026-09-26 01:59 UTC之前完成审核才能进入投稿版；这里不核实会议官方截止日期。

| 实验 | 完成审核 UTC | 投稿冻结前 | Plan SHA256 |
|---|---|---|---|
| 1 | 2026-09-25T16:38:28.371280+00:00 | True | `80aedd973ac71826f216e74648cb20ba21ccb0c00ff939b3b9bca078e2a0e798` |

原始逐token logprobs、token IDs、更新量和收据在 Lambda `/lambda/nfs/dami/hss/sink-next-20260925/exp*/`；轻量副本在本地 `results/sink-next-20260925/`。完整10方向明细见各项 [JSON](sink-next-20260925/)。
