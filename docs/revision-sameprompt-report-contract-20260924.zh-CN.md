# 同题多次生成：统计、报告与独立审计

本阶段接入已经冻结的[研究协议](revision-sameprompt-plan-20260923.zh-CN.md)。不修改采集、读出、问题划分或主比较；新文件只读取完成且通过独立审计的预测。

## 统计口径

- 主视图固定为 block28 / pre-final RMS / prefix16。主比较是每道混合测试题的 `AUROC(mean16) − AUROC(baseline)`，题间等权；同时保留 duplicate-current 对照。单类题不伪造 AUROC。
- 全部64道测试题保留在覆盖表；分别列完整四条回答的全对/全错/混合，以及每个视图可用轨迹、过短响应、唯一前缀序列数。相同前缀的多次生成不代表多份不同编码。
- 辅助跨题 AUROC/log loss/Brier 按轨迹汇总，但 bootstrap 的单位仍是问题。重采样所有64道测试题，包括无可用前缀的题；无法估计的重采样不伪造指标，保留实际有效重采样数。
- 每项使用固定 seed42 的2,000次问题bootstrap、逐项95%区间，未校正多重比较。主题内统计条件于本次可估计的混合题；辅助指标和监测区间条件于固定读出/校准阈值，不覆盖完整重训或重新校准的不确定性。
- `common_prefix_cohort.json`另外保存16/64均可用的同一批轨迹上的结果，仍保留64题覆盖分母。不同覆盖的辅助结果不能直接解释为时间趋势。

## 两边界监测

仅使用block28的16/64两个读出。每种方法在正确calibration响应的两边界最大风险上选一个严格 `risk > threshold` 阈值，使经验FAR不超过10%。没有可用边界用负无穷；没有正确校准回答则阈值与监测结果明确不可估计。阈值以有限数、负无穷或不可用三种合法JSON类型保存。

报告测试实际FAR、失败检测率、首报位置、潜在剩余token数及比例，并分别列错误/正确回答的潜在token影响。测试集不重新调阈值；相同校准目标不意味着相同测试FAR。节约量只是对已经完成的生成做假设停止计数，不能称为实际计算节约或正确率提高。allmean仅有64步辅助读出，不把它伪装成双边界方案。

## 产物和复用

根目录 `/lambda/nfs/dami/hss/revision-sameprompt-20260923/report`：

- `summary.json`：全部视图指标、校准阈值、监测结果和覆盖。
- `predictions.parquet`、`within_question.parquet`、`question_coverage.parquet`、`alarms.parquet`：完整预测、逐题AUROC、覆盖/前缀多样性、首次报警。
- `selection.json`和`source_manifest.json`：只按tuning选C的记录、数据来源SHA、权重位置。所有权重/候选/训练变换仍在`readouts/`，不复制大张量进报告或Git。
- `questions/`：全部224题原题、完整模型输入、参考答案、每题四条生成及其预测/报警。
- 两组PNG/PDF图与HTML索引；实际科学报告产生后必须人工式视觉检查，不能拿合成测试图当结果。
- `_SUCCESS.json`表示报告构建完成，**不等于统计审核或视觉检查完成**。独立`statistics_audit.json`绑定该报告收据，队列末状态仍明确`visual_review_pending`。

`audit_revision_sameprompt_report.py`不导入报告统计实现，独立使用秩统计与加权ROC、按题重采样、枚举可行阈值，重算主/辅助结果和区间，核对覆盖、所有预测、原文及报警计数。合成测试包含ties、无可用前缀、单类、负无穷阈值、无正确校准回答，以及修改结果后重封SHA仍被独立算术审计拒绝。

一个独立CPU follower等待现有readout队列完成并审计，再构建报告和运行独立统计审计。它不重启采集、不重复拟合、不调用GPU；失败保留partial目录，需要检查后使用明确恢复方式。运行源码SHA冻结，部署仍通过GitHub。
