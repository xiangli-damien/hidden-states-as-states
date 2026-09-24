# 一个有限的功能案例：序列下标与字母能否分离

## 问题来源与范围

MATH validation `math_2526` 的前缀是 `a_1, a_`，下一token `2` 在原模型中几乎确定，在local8重构后显著下降。它支持检查记号续写，不能证明数学推理变量已被定位。本实验只问：**把 donor 的计数进度写入 recipient，是否能保留 recipient 的字母？**

这是一个新建的、单一模板的合成控制任务。它不是MATH正确率、自然回答轨迹、通用记忆能力或原失败memory/modulo toy的扩展搜索。不会增加模板、层、rank、强度、样本量以追求成功。

## 固定任务与配对

模型按要求输出7个连续下标项，例如 `a_1, a_2, a_3, a_4, a_5, a_6, a_7`。在受控条件中明确提供合法assistant前缀 `a_1, a_2, a_3, a_4, a_`，从第5项下标开始自由生成。**前缀是人为提供的，不声称它是本轮自然生成的。**

recipient和donor的字母与起始下标都不同。例如recipient从a_1开始、donor从b_4开始：反事实目标是自由生成 `8, a_9, a_10`。它不同于原任务应输出的 `5, a_6, a_7`。这测试计数进度的交换，不把违反原输入要求叫“正确率提升”。

全部26字母×5起点形成130种clean上下文。65个不复用clean上下文的固定配对，哈希选64个，再固定16验证、48测试。方向由哈希预先决定；donor选择不读模型结果。没有跨pair重复clean上下文，不根据效果筛题；失配不补充新样本。

真正tokenizer先核对供给前缀至少16token，recipient/donor的token数量及去掉字母/数字后的token结构签名一致。patch始终只位于assistant前缀末尾16个或4个token，不进入用户prompt。各位置按尾部对齐，实际ID和位置入清单。

## 能力门槛

先只执行16对validation的两个side，每个side做两次无干预生成：

1. 只给原prompt，能否自由生成正确完整7项。
2. 给合法供给前缀，能否自由补全正确剩余项。

两类各32个clean上下文都至少29个正确，且至少12/16对满足双方两类正确与token对齐，才允许进入测试阶段。严格解析允许空白，拒绝额外解释、错误项数、额外格式、错误字母或下标。不达门槛就报告失败，**不运行test、不修改模板或阈值**。

测试阶段所有48对也做相同能力测量。主分析仅针对双方能力通过且对齐的pair；同时给出全部48对中的覆盖率，以及将不合格pair计为“未证明成功”的覆盖调整率。不能只展示成功样本。

## 干预定义

Qwen2-7B-Instruct固定revision，block14；主width16、辅助width4；greedy、repetition_penalty=1.0、max_new_tokens=64。这是原始argmax协议，区别于旧MATH采集继承的1.05。每条件fresh cache，一次forward hook；自由生成全部保存。

冻结原MATH GMM K64区域中心、local8和shared8方向，无新拟合或归一化。每个recipient位置的区域归属只由其原激活决定：

`h'_R = h_R + P_(s_R) (h_D - h_R)`。

它把donor投影到**recipient的同一组坐标**，保留recipient正交补。不能把两个不同簇的8个坐标直接相减。shared8使用完全相同的操作，仅把投影换成固定共享基。四种条件为identity、full_donor、local8、shared8。

这里的8维是传入donor变化的维数，recipient的完整状态仍然保留；它不同于只保留8维的压缩重构。测试能力通过后最多48对×2宽度×4方法=384个真实patch条件；validation只做能力检查，不用它挑干预参数。

full_donor指在同一block和选定位置替换完整hidden vector，不代表交换整套KV cache或模型内部状态。记录理想与实际bf16位移、区域分配、捕获一致性和一次hook。主比较local8−shared8同8连续坐标，但局部基存储更多参数；不假装所有预算相同。

## 判分与统计

- 主指标：三个续写下标等于donor对应进度，且新生成第6、7项仍使用recipient字母，格式和项数正确。
- 第5项的字母在供给前缀里，**不计为模型保留了非目标信息**。
- 同时报告目标下标序列达成、自由生成字母保持、格式失败、原recipient任务保持和覆盖率。
- 主对比：合格test pairs上width16的local8−shared8联合成功率；问题对为单位的配对bootstrap。其他方法、width4、分指标均为辅助。与full_donor比较如实呈现，不要求local胜出才报告。
- 不把token当独立样本；不因区间跨0继续增样本。单模板成功也只证明这个有限续写任务，跨层和更自然任务另需证据。

## 执行依赖与当前状态

先完成已授权的Llama GSM8K采集及公平FA/MFA队列；此实验不得抢占GPU。CPU准备实际通过64/64对齐检查，前缀均18token。4项数学/配对/判分/失败门槛测试在本地与Lambda通过。准备计划SHA为 `00b32a8d6cb46f3474f35eb92cbc5c88a0b595e9b74fd332b5f94cb919256b6d`。

执行器、独立原始审计、报告和独立统计核验均另行实现。队列应依次等待公平比较审计完成与GPU空闲，执行validation能力检查、CPU独立审计；过门槛后才执行test，否则科学性停止并报告失败。输入准备或随机小模型hook测试都不等于7B能力门槛通过，更不等于steering成功。

配置：`configs/revision_index_interchange_20260923.json`。拟用结果根：`/lambda/nfs/dami/hss/revision-index-interchange-20260923`。任何偏离已冻结配置的修复必须保留旧证据，并先说明差异。
