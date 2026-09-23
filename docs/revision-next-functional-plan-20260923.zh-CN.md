# 从局部重构转向明确功能：本轮追加工作

## 当前任务顺序

1. GSM8K Llama test 1,319 条继续采集，健康作业不重启。Qwen test 全量已完成、已审计。
2. 已部署的公平 FA/MFA 队列等待采集审计通过、GPU 释放，自动执行 2 题 38 条件 smoke、审计、128 题 2,432 条件、审计、报告和独立统计核对。固定 rank8 主比较，rank4/16 辅助，不扩大搜索。
3. **现在并行进行 CPU 输出差异分析**，直接使用已完成 MATH/GSM64 locality 的 logits 和逐参考 token 损失，不占 GPU。配置 `configs/revision_output_profile_20260923.json`；新结果根 `/lambda/nfs/dami/hss/revision-output-profile-20260923`。
4. 从差异画像提出一个有明确反事实目标的功能候选，再冻结独立测试协议。不能先用测试结果选案例，再把同一批结果叫确认实验。

## 输出差异画像的固定范围

- 同一模型 Qwen2-7B-Instruct、block14、生成16 token 边界；历史 MATH validation32/test32 和新 GSM64 分开。
- width16 的 identity、center、local8、shared8、shared64、clean remove-local8(alpha1)。四角表完整保留，KL 不拆成可加功能贡献。
- width1/4 仅使用真实存在的 identity、center、local8、shared8；不伪造 shared64 的窄窗口。扩大窗口同时扩大扰动能量，不能单独解释为更多位置的信息增量。
- 首参考 token、前16、2–16、16之后、完整参考损失，先每题平均再跨题平均。这里首参考 token 是回答第17个，不是 prompt 后的第一个。
- 全词表概率差、总变差距离、带符号 KL 项；最高概率候选、最大绝对概率变化和正负 KL 项逐题可读。数字、运算符、LaTeX、字母、标点等分类只是字符规则，不能直接命名计算功能。
- local8−shared64/shared8 的逐题差值、均值、配对问题 bootstrap、正差题比例、中位数、最大10%题占正差总量。删去大效应题的均值仅为探索性敏感性，不替换原结论。
- 原输入和输出 SHA、完整预期条件覆盖、实际参考 ID、逐 token 损失聚合、完整词表概率/类别守恒均核对；保留原文和所有选中条件。

## 对来稿建议的两项更新

来稿使用的是较早 MATH 32 道历史测试的数值。新 GSM64 确认已经完成：同8坐标 local8−shared8 的 KL 和完整参考 NLL 都改善，不能把旧 MATH KL 跨零概括成全部最新证据。另一方面，GSM64 的 shared64 MSE 比 local8 低；旧 MATH 的 MSE/KL 排序反转不能声称已在新目标上独立复现。

所有 shared 表示仍保留区域专属中心，shared64 优于 local8 不等于分区无用。下一 token KL、参考续写保真、任务正确率、选择性控制分开报告。

## 明确功能案例的进入条件

先从 **MATH validation** 原文和概率差提出一个范围小、可预先指定正确反事实的变量。其余现有题可描述性查看，不能充当新候选的独立确认。能力检查应记录所有候选题、基线能否完成、合格覆盖率；不无限扩大旧失败 toy 模板来寻找成功。

冻结 recipient/donor 配对规则、目标变量、非目标内容、位置窗口、唯一主指标、样本数和判分器，再用独立题测试。至少比较原 recipient、完整 donor 激活替换、局部坐标交换、相同坐标预算共享方向交换；保留参数预算差异。目标达成率、非目标保留率和整体覆盖率都报告。完整 donor 也不一定选择性成功，不能只留成功样本。

此协议和真实自由生成尚待实现；当前 replacement 的教师强制 KL/NLL 不能代替它。当前不能保证新功能案例与 2,432 条件 replacement 同时完成。

## 其他线

同题多次生成的 mean 增量、prompt/当前状态/长度/熵控制和 matched-FAR 是独立应用线。继续按原协议推进，不能自动连接成 local8 机制。跨层功能迁移放在功能变量明确之后。现有随机、径向、Gram 对照归档，不继续无边界增加 rank、seed、SAE/VQ 或分类器。

研究背景：[端到端稀疏字典工作](https://arxiv.org/html/2405.12241v2) 已区分 activation MSE 与输出保真；[Directions to Regions](https://arxiv.org/html/2602.02464v1) 使用区域及局部变化。当前新分析应回答具体缺失信息，而不是仅把 MSE 与 KL 不等同作为新理论主张。

## 每日 Git 保存

用户授权每天一次。现有半小时 heartbeat 在纽约时间21点后的首次运行检查 HSS/OpenAct/collection side branch，按本地日期去重。只有实际完成、验证的本任务改动才提交，作者使用已确认的 Xiang Li GitHub noreply；无改动不创建空提交。数据与权重留在数据存储，GitHub 保存源码、配置、测试和真实研究记录。重要修复仍即时提交。
