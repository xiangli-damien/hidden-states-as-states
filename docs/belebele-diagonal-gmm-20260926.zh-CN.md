# BELEBELE 两模型对角 GMM

2026-09-26 用户明确取消 Qwen MATH reliability，优先拟合刚完成采集的 BELEBELE。Reliability PID505843 已于05:45UTC收到SIGINT并退出，状态为 `cancelled_by_user`；原协议、部分候选和取消前状态均保留，不再启动。此任务不做reliability、MFA、预测器或steering。

## 输入与拟合口径

- 模型：Qwen2-7B-Instruct、Llama-3.2-1B-Instruct。采集各2700条；英、德、中各900条。
- 默认三语言合并，每个模型拟合一张逐层地图；保留语言元数据，方便比较同一地图的语言分布。语言是否分开已向用户询问；未收到不同指示时按合并执行。
- 输入是完整回答的原始 token mean；不做标准化、PCA或额外归一化。模型本身的最后RMSNorm保留，最终pre-RMSNorm另作一层视图。
- Qwen存储层0–28加最终pre：30组；Llama0–16加最终pre：18组。层0是embedding。
- 全部2700条参与无标签拟合，包括答错和达到长度上限的有效回答；正确性标签只作为元数据，不能称作独立测试AUROC。未来跨语言预测应核对平行题目，避免相同内容跨训练／测试泄漏。
- 每个K三次独立D²初始化；对角GMM，GPU float64，max_iter2000、tol1e-5、reg_covar1e-6。数值稳定用的内部平移恢复到原坐标，不改变尺度。

## K选择与输出

初始候选为1–16逐整数、20–80间隔4。若无容差最优K≥72，扩展96/112/128/144/160。围绕ICL前三名和0/2%容差选择做两轮整数细搜。只允许收敛候选参与选择；全部失败与非收敛记录保留，不悄悄改阈值。

ICL = BIC + 2 × 后验熵，越小越好。容差t选择满足 `ICL <= best + t * max(abs(best),1)` 的最小K，分别保存0%与2%。这是已搜索候选中的选择，不保证未搜索整数上的全局最优。报告任何搜索上界命中和空硬分配成分。

复用已验证的MMLU拟合、导出和独立审核实现，仅把输入路径、数据集名称和收据门槛用于BELEBELE。先Qwen后Llama，同一GPU顺序执行。

- 配置：`configs/belebele_diagonal_gmm_20260926.json`。
- 入口：`scripts/run_belebele_diagonal_gmm.py`。
- 结果根：`/lambda/nfs/dami/hss/belebele-diagonal-gmm-20260926`。
- 激活来源：`/lambda/nfs/dami/openact/runs/belebele_full_20260925/full/{qwen2,llama32}/belebele_{en,de,zh}`。
- 环境：`/home/ubuntu/hss-venv/bin/python`。
- 候选：`{model}/candidates/{view}/layer_XX/k_XXX/`，保存三初始化、最佳收敛参数、后验概率、MAP与最近中心分配。
- 选择：`candidate_metrics.csv`、`selection.csv/json`；图为`k_by_layer.png`。
- 地图：`exports/post_icl_0/`、`post_icl_0.02/`及两个`pre_final`版本，可由`hss.results.Result`加载。
- 跨层编号沿用cosine Hungarian、eta0.6，不视为跨层功能同一性的证明。

## 启动与完成门槛

采集05:20:51UTC完成，根_SUCCESS和5400覆盖均确认。GPU启动前再次审核每格900唯一ID、全部标签、源数据及所有被拟合均值块的SHA；把来源收据冻结。检查reliability已取消且GPU没有其他计算进程。先运行CPU/GPU同参数数值预检，再正式拟合。

各模型拟合完成之后必须通过独立审核：候选重启元数据、ICL计算与选择、每份导出的完整哈希、主要选择的参数／后验／恒等预处理。两个模型均通过后才写根COMPLETE。状态文件为`queue_status.json`，不是`job_status.json`；运行PID以`launch.json`为准，不重复启动。

本地针对性检查16通过、3个torch测试因本地未安装torch跳过；后者必须在GPU环境验证。代码经GitHub push → Lambda fetch/ff-only部署，不上传源码副本。初估两模型1–2小时；以实际首批完整层耗时校准，不能按准备或载入耗时外推。
