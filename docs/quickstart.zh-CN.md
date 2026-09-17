# 在 Lambda 使用 HSS

OpenAct 负责生成、采集和原始标签；HSS 负责读取激活、聚类、实验评估和出图。两套代码、环境和产物各自独立。

## 先看哪些文件

- [架构与公共接口](architecture.md)：模块边界、输入输出、结果格式、缓存和扩展方式。
- [论文图表与补充实验](paper-artifacts.md)：Figure 1–12、Table 1–3、各类 reliability/robustness 检查的对应入口。
- [数据目录](../configs/lambda/datasets.toml)：换数据位置优先修改这里，实验算法不需要改。
- [参数说明](configuration.md)：TOML/JSON、继承、单个参数覆盖、grid 的写法。

## Lambda 目录

代码在 `/lambda/nfs/dami/hidden-states-as-states`，新工作区在 `/lambda/nfs/dami/hss`。

| 目录 | 内容 |
|---|---|
| `studies/paper` | 实验配置、依赖、各任务执行状态 |
| `results/trials` | 正式实验的完整结果，按 trial ID 保存 |
| `cache/fits` | 可复用的拟合模型、预处理和候选 K 缓存 |
| `figures/paper` | 论文图、作图数据、参数与来源记录 |
| `figures/controls` | 稳定性、敏感性、消融检查 |
| `reports` | 数据量、完整性、缺失实验清单 |
| `validation` | 小规模验证结果；与正式实验分开 |
| `/home/ubuntu/hss-cache` | 本地磁盘上的激活 mmap 缓存 |

原来的 `hss-benchmark-*`、`hss-experiments` 保留，通过工作区的 `workspace.json` 查找。OpenAct 的采集路径和运行程序保持原样。

## 常用流程

```bash
ssh gpu2
cd /lambda/nfs/dami/hidden-states-as-states

# 1. 查看数据是否齐全、支持哪些聚类方法。
.venv/bin/hss data configs/lambda/datasets.toml
.venv/bin/hss methods

# 2. 正式实验配置已经生成。先检查 default_map 的规模和资源预算。
.venv/bin/hss plan /lambda/nfs/dami/hss/studies/paper/default_map.json

# 3. 运行一个任务；以后重复同一命令即可继续或复用已完成结果。
.venv/bin/hss suite /lambda/nfs/dami/hss/studies/paper/suite.json --only default_map

# 4. 校验结果。
.venv/bin/hss results /lambda/nfs/dami/hss/results/trials --validate --full

# 5. 从结果重画图，不会重新采集、加载模型权重或拟合聚类。
.venv/bin/hss figures /lambda/nfs/dami/hss/results/trials \
  --suite /lambda/nfs/dami/hss/studies/paper/suite.json \
  --destination /lambda/nfs/dami/hss/figures/paper \
  --only figure_03 figure_04 --max-trajectories 5000
```

不要反复生成同一个 study 来续跑；使用 `suite`。修改科学参数后可以新建 study。已有 study 需要显式 `--overwrite` 才允许重新生成，避免覆盖手动调整过的配置。

## 换方法、换数据、做 grid

GMM、MFA、KMeans、MiniBatchKMeans 共用结果接口。比如把已有配置改成 MFA：

```bash
.venv/bin/hss run configs/smoke.toml \
  --set 'cluster.method="mfa"' --set 'cluster.rank=4'
```

大规模搜索使用 `sweep`，并设置 `workers`、`threads_per_worker`、内存预算。已有 `configs/mfa_grid.toml` 可作模板。改变 eta 等下游参数会复用已有拟合；改变 seed、PCA、rank 则需要相应的新拟合。采集仍在运行时，当前默认使用 CPU 做 HSS；CUDA 后端可在资源空出后使用。

其他数据采集器可以导出 NPY + Parquet + `dataset.json`，再设 `data.source_format="arrays"`。OpenAct 原生支持 mean、prompt_last、prefix、sentence、tokens；`pre/post` 只切换最终 decoder layer 的 final RMS 前后位置。

## 结果可以怎样复用

一次完成的 trial 保存离散轨迹、元数据、划分、聚类模型、投影、标签统计、预测分数、评估、配置和校验和。把整个 trial 复制到另一台机器后，仍能读取模型、检查结果和重画图，不依赖原始 TB 级激活或拟合缓存。连续分类器的已保存分数可用于重新评估/画图；其 sklearn estimator 当前不做序列化，不能把这一点当作支持任意新样本的完整 probe 部署。

每张图另外保存 CSV/NPY、输入 trial ID、绘图参数、样本顺序和文件哈希。`index.html` 是图表索引；`coverage.json` 会说明缺哪些实验。用 `--check --strict` 可以只检查论文图表依赖是否齐全。

## 已验证与尚未完成

小规模验证检查的是代码和数据流程。全量 Llama-3.2 MATH 已能完整加载，但这不代表已经按全量 K=2…80 跑完所有模型和论文实验。

全量数值复现仍取决于未完成的采集、论文未给出的精确子集、Safety 数据差异，以及目前缺失的 prefix entropy/logprob。正文和附录的 FAR 定义也不同，两种定义均已提供显式配置。各项差异见 [论文复现条件](reproduction.md)，没有把缺失实验替换成合成结果。
