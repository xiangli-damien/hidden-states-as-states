# 同题多次采样：采集已启动

实现遵循已冻结的[研究协议](revision-sameprompt-plan-20260923.zh-CN.md)，未改题目、划分、采样参数或假设。

## 已运行的检查

- 3项问题选择/划分/seed测试、4项边界/均值/熵/恢复文件测试通过。
- 随机小Qwen在CPU上的真实前向和采样检查通过：未来token不影响已选前缀，捕获后的lm_head结果匹配完整模型，RMSNorm关系一致，每条seed不依赖此前生成顺序，生成前后同prompt状态完全相同。
- 真实Qwen2-7B冒烟2道训练题×4次生成，126.17秒；全部8条由CPU独立重新判分，token/角色/边界/均值/熵/margin/SHA及RMS关系通过。最大RMS关系相对误差9.60e-5。冒烟只验证执行，不用于报告准确率或mean效果。
- 正式896条采集随后自动启动，已通过审计的8条直接复用。每题共用一个有独立收据的prompt状态，每条轨迹使用固定seed；完成记录及其引用文件都通过SHA检查后才复用。

## 运行入口与数据

根目录：`/lambda/nfs/dami/hss/revision-sameprompt-20260923`。

- `plan.json`、`inputs.json`：原冻结224题、896个seed；准备计划SHA `4cc1f67b0e37ed99ebbc8f2125e848afc5b75d05c5affe6ed38e64ee2bb09adf`。
- `execution_plan.json`：实际执行源码、软件版本和原MATH判分器身份；SHA `18419a5bb1354f95a975749a15e5a376dfdb7e4df559b64bbe3870dcfec82028`。
- `generation_config.json`：实际解析的生成设置，SHA `3a550f7b535be356894a2782ed4dc0c29e4849e88dd5f2ab037058801e2715f7`。
- `model_snapshot.json`、`rmsnorm.npz`：模型层数/宽度、最终RMS的gamma/epsilon、版本。
- `smoke/`：8条原文、表示、收据和`audit.json`；不覆盖失败或原始输出。
- `full/`：完整采集持续落盘。每轨迹JSON保存完整回答、IDs、新标签、结束原因及文件SHA；NPZ保存实际prefix token表示、均值、当前向量、RMS后表示、完整下一token logits/logprobs。
- `queue_status.json`、`collection_status.json`：以实时值为准。初始队列PID449239，正式采集子进程449312；不要重复启动。

采集源码提交`1960378`，使用OpenAct环境；HSS环境用于后续CPU读出。全量写在dami，预计约5–6GiB表示与元数据，另加读出缓存。这个预算来自真实冒烟的约49.6MiB，按896条外推；不等于以前所有层、整段生成激活的存储需求。现在没有重新搬运已完成的全量MATH/GSM8K数据。

## 后续读出代码与验证

`fit_revision_sameprompt.py`实现固定3个视图、10种“视图×表示”条件，每种仅3个L2候选。没有训练新的聚类模型。所有权重、每个C的预测、训练/tuning ID、变换参数、最终选择均保存。只用train拟合变换和权重，tuning选C；calibration/test标签不进入选择函数。初次2,000迭代到上限时，仅允许再继续2,000次；异常收敛警告不会被当成已收敛。

`audit_revision_sameprompt_readouts.py`独立从真实raw prefix恢复特征，再重算train均值/方差、类别列、各候选预测与tuning log loss，并核对各角色的题ID。选择损失统一在float64计算，避免不同库的float32汇总误差影响审计。

额外5项题内AUROC/问题bootstrap/严格FAR/首次报警测试与2项读出选择测试通过。一个独立的低维合成端到端测试覆盖224题/896轨迹、全部30个候选和独立审计，连同前述测试共15项通过。**合成测试不是LLM实验结果。**

`run_revision_sameprompt_readouts.py`可以作为CPU后续队列：必须等完整采集与raw audit通过，再拟合及独立审计。启动信息以执行状态为准，不把代码存在当作队列已经启动。

## 仍未完成的部分

正式896条采集尚未完成，尚无本轮读出、题内区分或监测效果结论。2026-09-24 00:49 UTC快照为351/896，GPU单进程约15/40GiB。不要提前报告mean有增益，也不要把只有两个边界的检查称为逐token监测。

## 统计与报告已接入自动队列

源`2996044`补齐了结果汇总、全部224题原文页面、两组PNG/PDF及独立统计审计。本地和Lambda共18项测试通过；额外完整2,000次bootstrap的合成报告/独立审计在本地用时28.81秒，合成图仅用于布局检查，不是LLM实验结果。具体统计口径见[报告契约](revision-sameprompt-report-contract-20260924.zh-CN.md)。

原CPU读出队列PID449467仍等待全量采集与raw audit。新增且唯一的CPU报告队列PID449932已启动，等待读出拟合与审计，随后自动构建报告并独立重算统计。它不重复拟合或调用GPU。状态见`report_queue_status.json`，冻结源码见`report_queue_plan.json`。成功末状态仍是`statistics_audited_visual_review_pending`，之后还须检查真实图表、PDF、链接与原文，再交付结果。
