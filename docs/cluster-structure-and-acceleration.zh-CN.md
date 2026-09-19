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
