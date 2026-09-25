# 冻结 prompt-last 地图的逐 token 迁移

## 固定问题与范围

用户要求把后续每个 token 放入同一张地图，看下一 token KL。先固定旧 cases.json 中各数据集前三题，无新结果筛选：MATH math_3992/2916/2668，GSM8K train_3094/156/3216。共 6 题、1,873 个预测位置，覆盖这些原记录完整回答，不截断到前 16 个 token。示例范围，不作总体正确率推断。

复用 promptlast-replacement-20260924/decoder.npz，不重新拟合：Qwen2-7B-Instruct、block14 输出、raw 3584D、最近中心路由、K32、共享／局部 rank8。记录模型与所有输入、代码 SHA。沿用原始问题和原生成 token，不进行新采样。

## 位置、操作与比较

- p=0：输入 prompt-last，预测原回答 token1。
- p=1：输入原回答 token1，预测原回答 token2。以此类推，直到预测原记录最后一个 token；不在 EOS 之后预测。
- independent：每次复制未修改的 clean KV cache，只替换当前位置，分支完成后丢弃其 cache。
- cumulative：从 prompt-last 开始每个位置都替换，各方法保留自己的修改后 KV cache。
- 六个条件：两种历史处理 × centroid / shared8 / local8。另有原模型 identity。
- 所有方法每一步读同一原回答前缀，重新用当前实际 block14 向量找中心。没有用 future mean、标签、固定首簇或自由生成后不同前缀比较。

同一层输出处干预、固定 token 序列下，当前层以下计算未改变，所以两种历史方式的待替换向量、路由和重构向量应相同；后续层的 cache 不同，KL 可以不同。每一步执行该不变量检查。

## 保存与验证

每 32 个位置保存所有 7 条完整词表 float32 log-probability，以及真实／替换向量、目标 token、region、距离、KL、NLL、几何误差。每题可恢复边界清晰，部分失败不静默重用。所有完整原始数组落 Dami，新目录，不修改旧实验。

随机 tiny Qwen 的逐步 cache 计算对照全前缀重放，覆盖独立／累积和 3 个算子；确认累计历史确实改变结果。真实模型先各 1 题 × 4 位置冒烟及独立数值审计，再运行 6 题全部位置。最后从保存的完整词表分布独立重算每一个 KL，检查所有位置、路由与 bf16 真实替换。

汇总先题内平均，再数据集内三题等权平均；不把 token 当独立样本给窄置信区间。导出完整逐位置曲线、原文与 CSV。KL 单位 nats，衡量输出保真度，不是准确率百分点或纠错能力。源码按本地提交 → GitHub → Lambda fetch/ff-only 部署。

根目录：/lambda/nfs/dami/hss/promptlast-tokenwise-20260925。配置 configs/promptlast_tokenwise_20260925.json。
