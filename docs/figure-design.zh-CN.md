# 论文图形重构：数据、显示和复现边界

## 实际阅读的参考代码

参考目录来自 Lambda 的 `/home/ubuntu/dami/code/llm-trajectory-dynamics/`。只读取代码和设计思想，不调用其中的拟合、载入或示例数据流程。

| 参考文件 / 函数 | 采纳的表达方式 | 本项目实现 |
| --- | --- | --- |
| `refracluster/experiments/clusterlens_paper_figures.py`: `compute_positions_ranked`, `add_curved_edge`, `filter_edges_smart` | 每层按全局 ID 排列、列居中、弧线、按占用调整节点、按转移计数调整边 | `hss.viz.publication.state_graph` |
| 同文件 `add_outgoing_entropy`, `plot_cluster_dynamics_unlabeled` | 暖色归一化流出熵，末层无定义为灰色 | `graph_display`；原始 bits 保存在 nodes.csv |
| 同文件 `plot_cluster_dynamics_labeled_relative`, `compute_flow_table` | 蓝—白—红相对正确率；边也按实际成员正确率着色 | nodes/edges 的 accuracy_delta + 同一显示范围 |
| 同文件 `plot_heatmap_diverging`, `compute_birth_death_layers` | 缺席状态留白、首次/末次出现三角标识 | `correctness_bands` |
| 同文件 `plot_rel_icl_surface` | magma_r 曲面、选 K 路径、容差轮廓 | `icl_surface` |
| `refracluster/experiments/02_sta.py` | 频率/累计占用、轨迹相似度、逐层动态分面 | `geometry` |
| `refracluster/experiments/01_main_figure_stability.py` | 将不同扰动和不同指标分面，并保留基线 | `stability_summary`, `separation` |
| `cluster/plotting_pub.py`, `cluster/alllayer.py`, `cluster/standard.py` | 紧凑论文版式、图例与曲线分离 | 共享 `style.py` 与辅助图 |

已逐页视觉对照论文 Fig. 3–12，尤其状态图、三联几何图、稳定性、相对 ICL 和正确率热图。

未采纳：`clusterlens_paper_figures.py` 的 `target_avg_k` 自动目标簇数设置；`01_stability_new_plot.py` 的 `generate_realistic_data()` 模拟数据。不能通过这些设置使图形看起来与论文相同。

## 哪些差异是画图，哪些是实验

1. **表示方式已修正**：之前编号弱、边直且杂、稳定性把所有 trial 对混在一起、ICL 缺少可读路径、网页把图藏在折叠项里。现在主要论文图按顺序直接可见，方法各自有完整图集。
2. **统计定义必须明确**：图中的归一化熵为 `H_bits / log2(实际观察到的后继状态数)`；只有一个后继时为零，末层为缺失。归一化在筛边前计算。所有边的原始计数仍保存。
3. **显示筛选不改变数据**：默认显示 count≥10、P(outgoing)≥0.02、每源状态最多5条边。完整 nodes/edges 与 display CSV 分开；每张图注明保留的转移质量占比。所有节点均显示，不合并、不重编号。
4. **共享正确率色阶**：默认 ±0.75，零点是该 trial 的全局正确率。越界用色条扩展标明；节点/边正确性由真实标签聚合，缺失不填零。这不是显著性或因果图。
5. **ICL 色阶压缩只影响显示**：上限为有限相对值的95%分位数（下限0.05），完整值留在CSV；相对量定义保持 `(criterion-min)/max(abs(min),1)`。轮廓是配置的 parsimony tolerance。不改变已选择的 K。
6. **稳定性比较可解释**：每个 seed/子集 refit 对比同方法的原始拟合。均值和 min–max 阴影对应已有扰动，不伪称置信区间；原始所有两两比较仍保存。它检验固定 K 的中心/归属稳定性，不替代论文独立选 K 的稳健性实验。
7. **预测图体现类别不平衡**：AUROC + average precision；AUROC CI 为固定拟合模型、分层重抽测试回答的95%区间，不包括训练随机性。AP随机参考为测试集正确比例。准确率、多数类基线、balanced accuracy/MCC 均保留在 CSV。
8. **仍存在实验数值差异**：当前默认 GMM 有48个全局状态、215个逐层节点；论文 Fig.3 是12个全局状态、最大8簇。画图不能消除这项差异。当前最后层取 post-RMS，末层出现新状态也包含 RMSNorm 和最后 block 的影响。须进一步按采集/层语义、特征归约、选 K 及对齐协议逐项核对，不能通过丢状态或强设目标 K 伪造一致。

## 模块与复现

- `analysis/tables.py`：真实 trial → 完整派生表，不依赖画图配置。
- `viz/publication.py`：表 → 论文图，不读取 activation、不拟合。
- `viz/style.py` + `configs/figures.toml`：渲染参数；不改变 data/fit/trial 的来源标识。
- `viz/artifacts.py`：PNG/SVG、表、输入哈希、已解析参数和逐图 `visual_encodings`。
- `viz/review.py`：按论文顺序的 `paper.html`、各方法图集、逐题查看与覆盖表。

在 Lambda 仓库根目录执行：

```bash
HSS_FIGURE_STYLE=configs/figures.toml \
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nice -n 10 .venv/bin/python scripts/render_math_review.py \
  --study /lambda/nfs/dami/hss/studies/llama32-math-20260917 \
  --destination /lambda/nfs/dami/hss/reports/llama32-math-20260917-publication
```

省略 `HSS_FIGURE_STYLE` 使用与示例 TOML 相同的默认值。修改颜色范围或边阈值只触发重绘；marker 将已解析的显示参数纳入缓存键。若要显示全部边，设置 edge_min_count=0、edge_min_probability=0、edge_max_per_source=0。大数据的质量诊断仍使用已保存特征缓存；不会重新生成或拟合模型。

## 当前复现边界

当前是 Llama-3.2-1B × MATH 5,000 的真实数据适配。Qwen图不改称原论文数值；跨模型/数据图、独立K选择稳定性、未采集的token entropy/logprob基线不填模拟曲线。MFA若达到迭代上限但未收敛，仍明确保留收敛标志与似然轨迹。图形风格接近论文不等于数值复现完成。
