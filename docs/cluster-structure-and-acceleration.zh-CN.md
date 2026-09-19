# Qwen MATH：MFA 加速与已完成聚类的方差诊断

## 加速保持什么不变

`scripts/accelerate_cluster_study.py` 在独立目录续跑现有 CPU study。原始
5,000×3,584 response mean、所有层、K 网格、rank 网格、3 次初始化、float64
EM、2,000 次上限及 1e-5 收敛阈值均不改变。已有完成候选仍指向原始 fit 目录，
不复制或覆盖其参数；进行中的 EM restart 检查点复制到新 GPU cache。
`migration.json` 保存源协议、驱动哈希、导入候选、检查点哈希和执行设置变化。
一个 MFA GPU worker 与 OpenAct 共用显卡，启动拟合前保留至少 8 GiB 余量；
GMM、KMeans 和 ICL 指标计算继续使用 CPU。

原 controller 和所有子进程必须先停止；迁移检查会拒绝仍在运行的源进程。
迁移后先验证新 worker 前进，再退出原 CPU 进程。失败时可以恢复暂停的旧进程。
不要修改原 study 协议来冒充同一个执行后端。新实验记录 CPU/GPU 混合来源，
数值一致性不意味着逐位完全相同或所有初始化都必然走到同一局部最优。

代表性实测位于 `/lambda/nfs/dami/hss/mfa-speed-20260919/benchmark_existing.json`：
最后一层 K=21、rank=8，从同一保存参数执行 4 个 E-step，CPU 两线程 32.77 s、
GPU 1.23 s；最大平均 log likelihood 差 4.55e-13，noise 参数相对误差 6.69e-14。
这是一项与原有 CPU workers 并行时的短程测试，不能把 26.7× 当作整个队列提速，
也不包含初始化、读写、ICL 和与生成任务竞争资源的成本。

```bash
python scripts/accelerate_cluster_study.py \
  --source /lambda/nfs/dami/hss/qwen-math-ablation-20260918 \
  --directory /lambda/nfs/dami/hss/qwen-math-ablation-gpu-20260919
```

## 不等待消融结束，直接观察已有簇

```bash
python scripts/inspect_cluster_structure.py \
  --source /lambda/nfs/dami/hss/qwen-math-ablation-20260918 \
  --output /lambda/nfs/dami/hss/qwen-math-cluster-structure-20260919
```

报告包括：

- 同一个 PCA 显示坐标系下，KMeans / GMM / MFA 的固定 K=21 对照；另存
  KMeans silhouette 所选 K=2、GMM 最低 ICL 所选 K=51 的诊断。
- 每簇人数、正确率、回答长度、题型、簇内总方差、前 32 个经验主成分的谱。
- MFA 的 `trace(WWᵀ)` 与 `sum(ψ)`，以及与潜变量旋转无关的因子子空间重叠。
- 每道题原文、回答、正确性与归属，离线 HTML 可以按方法和簇选择。
- `structure.json`、`members.parquet`、`cluster_summary.csv` 和可独立查看的 PNG。

PCA 只用于画图，绝不改变用于聚类的输入。经验谱由 randomized SVD 估计，
总方差分母精确计算；协方差采用总体定义（除以 n）。经验谱使用 hard assignment，
MFA 模型协方差来自 soft mixture 拟合，二者不是同一个估计量。
所有正确率关联均为同批数据的描述统计；不能据此宣称泛化或因果机制。

## 交互式簇画像与局部结构

先生成上述原始诊断，再运行：

```bash
python scripts/enrich_cluster_structure.py \
  --report /lambda/nfs/dami/hss/qwen-math-cluster-structure-20260919 \
  --states /home/ubuntu/hss-cache/data/473f2199a8742bdf18996361/layer_28.npy \
  --rows /home/ubuntu/hss-cache/data/473f2199a8742bdf18996361/rows.parquet
```

脚本验证原始矩阵形状、逐行 sample ID 和各方法固定标签，不修改聚类或拟合参数。
`exploration.json` 保存全部派生数据、原始文件与标签哈希、抽样行号、软件版本、
投影参数。HTML 模板独立放在 `src/hss/reporting/templates/cluster_explorer.html`；
修改展示后使用同一命令的 `--render-only`，无需重算诊断。
`render_manifest.json` 记录数据、模板与渲染脚本哈希；`cluster_atlas.pdf/png`
是可导出的六面板共享投影图。交互图与导出图使用相等的横纵坐标单位。

- 全局 PCA 与三套共享 t-SNE：perplexity 30/80，seed 42/43，random 初始化，
  1,000 轮；t-SNE 输入为仅用于显示的 PCA50，无标准化，保留约 81.75% 方差。
  坐标不依赖方法标签或正确性。二维岛的面积、密度、间距不直接代表原空间。
- 在固定 512 个 anchor 上，精确比较对全部 5,000 条的原始欧氏近邻和显示近邻。
  Recall@15：PCA2 约 6.3%，三套 t-SNE 约 41.7–43.6%；trustworthiness
  为约 0.977–0.979。PCA50 的 Recall@15 约 76.8%。二维仍丢失大量局部关系。
- 题型富集为簇内占比 / 全体占比，显示原始人数，未做显著性推断；难度分布、
  长度分位数、截断率与 95% Wilson 正确率区间帮助检查组成效应。区间不包含
  拟合与模型选择的不确定性。组成校正仅减去同批「题型 × 难度」条件正确率，
  不是独立验证，也未控制长度等混杂。
- 每簇单独 PCA，用于比较主方向两端、距经验中心最近/最远的真实回答；
  不把中心附近样本称为精确 medoid。有效维度 `trace(C)^2 / trace(C^2)`
  由完整经验协方差计算，与拟合 rank、前 32 维累计占比区分。
- ARI / AMI 与交叉成员矩阵比较算法划分，不能代替重复拟合稳定性。
  原始欧氏 silhouette 使用相同 1,500 条固定子样本，对椭球混合和不同 K
  有不同偏好，不作为这次诊断的选模标准。

当前 MFA K=21/rank8 簇的经验有效维度中位数约 12.7，范围 6.8–24.6；
因而 rank8 之外仍有方差，这是 MFA 对角残余所允许的，不等于不收敛。
MFA 与 GMM 的 ARI 约 0.330，与 KMeans 约 0.282，说明方法仍明显影响划分。
例如 MFA Cluster 2 的中级代数富集约 3.71 倍、长度中位数 1,044 tokens、
正确率 5.38%（n=260）；题型与难度组成基线为 22.38%，差值仍为 -17.0 个
百分点。这个描述提示继续查原文和长度混杂，不能自动解释为“错误机制”。
