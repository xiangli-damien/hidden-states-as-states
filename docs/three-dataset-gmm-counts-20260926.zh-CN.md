# 三数据集 GMM 簇数：归一化层深对比

按用户要求，将已有 Qwen2-7B-Instruct 与 Llama-3.2-1B-Instruct 的 MATH、MMLU、BELEBELE 地图绘制成两张图：全部取0% ICL容差，以及全部取2%。每张图左侧Qwen、右侧Llama，三个数据集使用固定颜色。

## 口径

- 横轴为存储层号除以模型block数：Qwen为28，Llama为16。0是embedding，1是最终post-RMSNorm。仅对横轴归一化，没有重新处理激活或归一化K。
- 纵轴是所选GMM成分数K，不是簇内样本人数。为便于阅读，0%图纵轴0–120，2%图0–60；同一张图两模型纵轴一致。
- 均使用完整回答raw token mean。MATH全量5000、MMLU全量14042、BELEBELE英德中合并2700。
- 0%仍使用ICL，仅不使用容差；2%是在最低ICL允许的2%范围内选最小K。具体公式为 `ICL <= best + t * max(abs(best),1)`。
- 保留历史选择，没有重新拟合。三个数据集的样本数、K搜索集合及初始化预算不同，本图是现有结果的描述性比较。
- Llama MATH 0%的embedding层K=36旧拟合未收敛，红叉标示；其余展示点均收敛。未把独立的最终pre-RMSNorm结果混入曲线。

## 输出

结果目录：`results/three-dataset-gmm-counts-20260926`。

- `gmm_three_datasets_icl_00.png/pdf/svg`：0%。
- `gmm_three_datasets_icl_02.png/pdf/svg`：2%。
- `gmm_three_datasets_both_tolerances.pdf`：两页PDF。
- `plot_data.json`：全部276个点、逐点收敛标志、拟合路径及来源哈希。
- `index.html`：两图、下载链接及可展开的逐层数值表。

本地查看：[图与逐层表](http://127.0.0.1:8794/report/gmm_three_datasets/index.html)。该页面是轻量结果的副本，已有本地报告服务提供访问。

生成命令：`python3 scripts/plot_three_dataset_gmm_counts.py`。脚本检查旧MATH/MMLU图表与原选择文件的来源哈希，校验BELEBELE独立审核收据与选择文件哈希，再核对两模型每个数据集两种容差的全部层覆盖及绘图数组。两张输出图均已实际查看，文字与坐标未裁切。没有启动新GPU任务。
