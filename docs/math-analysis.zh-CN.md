# 查看完整 Llama MATH 实验

本 study 使用 Llama-3.2-1B-Instruct 的全量 MATH 5,000 条、原始 2,048 维激活、全部 17 个已采集层位置（含 embedding）。生成侧使用全回答 token 均值，prompt 侧使用最后一个输入 token；最终层采用 post-RMS。逐 token 激活仍保存在 OpenAct，此处的层间轨迹不是逐 token 时间轨迹。

## 方法与指标

- GMM：对角协方差，ICL 扫描 K=2–80；选最优值 2% 容忍区内最小 K，2 次初始化，最多 200 次迭代。
- KMeans / MiniBatchKMeans：既运行与 GMM 相同逐层 K 的对照，也独立扫描 K=2–80，以固定随机种子抽取的最多 2,000 条 silhouette 选 K（不使用正确性标签）。
- MFA：rank=8，与 GMM 使用相同逐层 K。属于控制簇数量的新增方法比较，**不声称完成 MFA 独立 ICL 最优 K 搜索**。
- 所有方法：报告 silhouette、Calinski–Harabasz、Davies–Bouldin、簇占用、单样本簇、状态转移及正确性/题目类型/难度关联。不同准则的数值不跨方法直接比较。
- GMM/MFA 保存 EM 收敛状态。达到迭代上限的结果可以查看，但不能称为已收敛。
- 固定 K 的 seed=43/44、拟合子集比例 50%/80% 检查中心余弦距离和 ARI。与重新选 K 的稳定性是不同问题。
- Prompt 预测采用按正确性分层的 40/60 划分；K、聚类、标准化和分类器只看训练组。HSS-NB 使用 Laplace 平滑，连续基线使用最后层。测试集 bootstrap 是固定模型的不确定性，不包含重新训练的变异。
- Prefix 监测按回答组 40/20/40 划分；10% FAR 阈值只在验证集校准。缺少的逐 token entropy/logprob 不用完整回答的指标替代。

## 运行与续跑

```bash
cd /lambda/nfs/dami/hidden-states-as-states
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/run_math_analysis.py \
  --directory /lambda/nfs/dami/hss/studies/llama32-math-20260917 \
  --stage core --workers 10
```

依次运行 `core`、`prediction`、`reliability`、`monitoring`；只有前一阶段的引用存在，后续阶段才能执行。同一命令会复用每个 K 的缓存。每个调度器使用 48 GiB 内存预算，并预留 32 GiB 可用内存；多个辅助调度器需要合计检查资源。GPU 不参与拟合。`study.json` 记录已完成 trial，`progress/*.json` 记录每层，`configs/*.json` 是完整执行记录。调度器会从 `--base` 重建配置，因此参数变更应使用新的 base/study；也可复制某个 JSON 后用 `hss run` 单独运行。改变科学代码后使用新 study，防止混入旧版本结果。

```bash
.venv/bin/python scripts/render_math_review.py \
  --study /lambda/nfs/dami/hss/studies/llama32-math-20260917 \
  --destination /lambda/nfs/dami/hss/reports/llama32-math-20260917
```

打开报告的 `index.html`。整个报告目录可以下载到本地，直接打开，无需服务器或第三方 JS。包含每题原文、完整模型输入（chat template）、回答、标准解答、抽取答案、标签和各方法的层间轨迹；文本使用 `textContent` 展示。模型输出不能执行网页脚本。图表为 PNG/SVG，指标为 CSV，原文另存 Parquet；每个图集有来源 trial 和输出哈希。

## 与论文的对应边界

可生成默认状态图、占用/轨迹相似度/层变化、标签关联、ICL 图、prompt/response 正确性状态图、状态带、方法对照、稳定性以及单模型预测/监测表。

论文中 Qwen 对应的图在这里是 **Llama-MATH 适配**。跨模型/跨数据集图和完整 Table 1 必须等对应数据到齐。本文档列出的是可执行协议；完成与否以实际 `study.json`、`coverage.json` 和结果 `_SUCCESS.json` 为准。

图集附 MFA 每次 EM 迭代的平均对数似然及变化幅度。未达到收敛阈值的拟合须标注，不能仅凭图形漂亮或似然较高来判断方法优劣。跨方法几何指标按相同逐层 K 比较，最终任务价值以独立测试集的预测结果为准。

最终层 post-RMS 与上一层的差异同时包含最后一个 block 和 RMSNorm，不能把新状态全部归因于语义变化。论文正确性状态带使用相对全局正确率的 ±30 个百分点阈值；全局正确率低于 30% 时不会出现 low 标签。逐状态 CSV 另给实际正确率、样本量、回答长度及截断比例，便于解释。
