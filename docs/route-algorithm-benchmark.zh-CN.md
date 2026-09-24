# 无标签路径结构：广覆盖算法比较

## 本轮结果

已完成 **210 个候选 + 16 个 HMM 续训**。正式结果使用 `state-route-algorithms-20260921-final`；初始版本及 `-refined` 收敛检查版本保留。训练3,011题，验证1,003题，测试986题。初始拟合18.9分钟，HMM并行续训与复制约34秒，另有评分一致性修复、评价和交付时间。

初始27个HMM中16个触及100次迭代上限，包括最终选中的三个地图×三个种子。按预先声明的无标签收敛问题继续41–250次迭代后，**27/27 HMM与27/27路径混合模型均达到数值收敛标准**。续训不使用正误标签；重新按验证NLL选HMM，要求该设置的三个种子均收敛。其他模型未重新训练。神经网络按验证早停／固定预算结束，不声称找到全局最优。

| 测试主分数，方向预先固定 | GMM | 同K GMM | MFA |
|---|---:|---:|---:|
| 节点罕见程度基线 | 0.440 | 0.637 | 0.635 |
| 高阶Markov | 0.566 | 0.502 | 0.521 |
| 路径混合 | 0.563 | 0.495 | 0.517 |
| 收敛HMM | 0.482 | 0.499 | 0.498 |
| GRU | 0.564 | 0.504 | 0.535 |
| 因果Transformer | 0.558 | 0.505 | 0.541 |
| 掩码Transformer | 0.559 | 0.502 | 0.543 |
| Deep SVDD | 0.561 | 0.530 | 0.534 |
| 后缀结构对比学习 | 0.467 | 0.484 | 0.495 |
| LOF | 0.580 | 0.539 | 0.586 |

共同基线：长度 **0.776**，平均token熵 **0.685**，二者验证百分位等权平均 **0.790**。完整表还包含kNN、Isolation Forest、One-Class SVM、PCA、DAE、VAE、固定集成，以及预设top4局部异常分数。上述最高值是事后描述，不是使用标签选出的可部署模型。

路径结构确实被学会：MFA测试集的独立节点NLL为1.979 nats/层，因果Transformer为0.456；GMM独立节点3.801，高阶Markov1.024。更好地预测状态分布，没有转化为更好地区分正误。

本轮最有利的固定组合来自MFA**节点**基线：在长度+熵上增加 **0.01094 AUROC**，点态95%区间 **[-0.00996, 0.03184]**，跨0；这并不是转移结构的增益。其长度×熵分组内AUROC为0.639，说明静态状态占用仍值得单独研究，但分箱不能排除题型、难度等其他解释。预设top4的最高值同样来自节点基线（同K GMM，0.642）。

结论范围：这些无标签密度、异常性和一致性目标尚未产生优于长度/熵基线的失败检测器。它不证明state sequence不含正误信息、不否定其他目标，也不把方向反转后重新包装为成功。

[本地交互报告](http://127.0.0.1:8775/report/index.html)。模型、逐层损失、逐题分数、候选诊断和986题测试原文均保存。

重载核验发现sklearn LOF在大量相同离散距离下可能随查询批大小改变并列邻居。最终版改用严格等价的one-hot欧氏距离（sqrt(2×不同层数)），以训练行号稳定处理并列，重算三个LOF。VAE评分保留4次积分，但使用固定共同Gaussian噪声，使同一条路径的分数不随查询顺序、批次或CPU/CUDA的随机数生成器改变；18个VAE仅重算分数，训练权重和验证选参不变。两项修复都来自可复现性检查，不按标签性能挑选。GRU的CPU/CUDA浮点差异可达约2e-4，不视为逐位一致。

## 范围与冻结规则

本轮覆盖 15 类方法，另有节点罕见程度基线与等权集成。“先进”不代表适合当前数据，也不声称穷尽全部算法或达到某个榜单的 SOTA。

主数据为 Qwen2-7B-Instruct × MATH 5,000；使用原 GMM、MFA 和与 MFA 同 K 的 GMM 共三张地图。输入是完整回答 hidden mean 的 28 层 local state ID，不是生成 token 时间序列。原 hidden state 不重新 normalize。one-hot、网络内 LayerNorm、PCA 基线均只作用于检测器输入或内部表示，不改变原聚类。

沿用 SHA256 canonical question-group 五折：0 测试，1 无标签验证，2–4 训练。训练器只读取 sample_id、question_group、state sequence。拟合、早停、选参、分数方向、组合权重、阈值都不用正误标签。所有分数落盘并冻结后才执行单独评价脚本。

**这仍是探索性实验：**原聚类中心、K/rank 曾在全体 5,000 题上拟合，且数据此前被反复研究。检测器留出测试集不能消除聚类地图的传导性使用，也不是独立新数据的确认实验。

## 方法与参数

统一配置：`configs/route-unsupervised.toml`。神经与EM方法三个种子 920/1920/2920，经典方法固定种子920；神经网络 width64/128，最多45epochs，patience7。小网络限制 PyTorch GPU 分配为设备8%，CPU2线程，单候选串行，保护原采集任务。全部候选保存，无标签验证集决定选参；不按测试 AUROC 选算法冠军用于部署。

| 类别 | 方法 | 设置与无标签选择 |
|---|---|---|
| 节点基线 | 独立层类别频率 | Laplace1 |
| 条件预测 | Backoff Markov | order1/2/3/4，strength10，验证NLL选阶 |
| 路径族 | Mixture of Markov chains | 路径族2/4/8，MAP平滑EM，验证NLL |
| 隐结构 | 层相关 categorical HMM | 隐状态2/4/8，层相关发射与转移，验证NLL |
| 邻域 | kNN | one-hot欧氏距离，k20 |
| 局部密度 | LOF | k35，novelty评分，训练行号稳定处理距离并列 |
| 隔离 | Isolation Forest | 400树，max_samples256 |
| 核方法 | One-Class SVM | RBF，nu0.1，gamma=scale |
| 线性重构 | PCA | one-hot中心化后32主成分，重构误差 |
| 非线性重构 | Denoising AE | 30%层状态置空；类别交叉熵；瓶颈width/4 |
| 概率潜变量 | Categorical VAE | Gaussian潜变量，CE+KL；4组固定共同噪声积分评价 |
| 自回归 | GRU | 严格右移输入，前缀预测下一层状态 |
| 自回归注意力 | Causal Transformer | 两层、四头、严格右移输入与因果mask |
| 双向条件预测 | Masked Transformer | 随机mask训练；逐层leave-one-out伪似然评分 |
| 深度单类 | Deep SVDD adaptation | 无偏置编码器、AE预训、固定中心、25epoch；固定width64主版本以免用收缩目标挑退化宽度 |
| 结构自监督 | Suffix-corruption contrastive | 真实路径与共享中间状态处交换后缀的路径；交换严格保留相邻边计数 |

神经方法为面向类别路径的透明实现，不冒充原论文官方复现。Deep SVDD 等单类方法面对的是正确与错误混合分布，正常假设不自动等于正确。HMM/MM 记录收敛标志与曲线；达到上限不伪称收敛。

三个地图各70候选，共210。七类神经模型和两类EM模型都含三个种子。主分数为逐层mean，预设top4作为局部异常敏感性分析。无标签验证集百分位的等权集成保持固定，不按正误调权。

## 评价与解释

1. 只在检测器测试集上评价失败 AUROC、AUPRC、风险覆盖、点态bootstrap区间。
2. 固定大分数指向失败，不因AUROC<0.5翻转。
3. 长度、完整回答平均token熵分别作为基线；二者的无标签验证百分位等权平均是联合基线。
4. 预设三者等权（长度、熵、路径）与联合基线比较，报告配对bootstrap差值。它不等同于最优监督分类器的增量检验。
5. 长度四分位×熵四分位内计算配对加权AUROC；另用无标签验证集拟合“长度/熵→路径分数”，报告测试残差AUROC。分箱和有限回归都不能证明彻底排除混杂。
6. 验证集90分位阈值报告观测测试FAR。它只校准异常标记比例，不能保证10%正确样本误报率。
7. 全部主方法高于随机的检验做BH校正。区间与组合差值为条件于当前地图和已拟合模型的逐项区间，不包含重新拟合不确定性；挑最佳方法仍需新数据确认。

## 保存与复现

远端最终结果：`/lambda/nfs/dami/hss/state-route-algorithms-20260921-final`。本地同步到仓库 `results/state-route-algorithms-20260921-final`。初始结果和HMM续训版本同样保留。

```bash
.venv/bin/python scripts/benchmark_route_unsupervised.py \
  --source /lambda/nfs/dami/hss/state-route-initial-20260920 \
  --output /lambda/nfs/dami/hss/state-route-algorithms-20260921 --device cuda
.venv/bin/python scripts/evaluate_route_unsupervised.py \
  --source /lambda/nfs/dami/hss/state-route-initial-20260920 \
  --root /lambda/nfs/dami/hss/state-route-algorithms-20260921
```

同一冻结协议允许从完整候选恢复。修改代码、配置或输入后必须使用新输出目录。

本次HMM续训入口为 `scripts/refine_route_hmm.py --source INITIAL --output NEW --data INPUT_STUDY`；会复制初始记录到新目录、继续未收敛候选、重新冻结分数。对新目录再执行独立评价脚本。完整性校验入口为 `scripts/audit_route_benchmark.py --root RESULT`。

最终评分修复入口为 `scripts/stabilize_route_lof.py --source REFINED --output FINAL --data INPUT_STUDY`，同时固定LOF并列处理与VAE积分噪声，再在FINAL目录执行评价。新跑的benchmark已直接使用这些修复，无需重复执行修复脚本。

新样本先使用**原来同一套聚类模型**得到各层local ID，保存N×28整数 `.npy`；然后：

```bash
.venv/bin/python scripts/score_route_samples.py \
  --benchmark results/state-route-algorithms-20260921-final \
  --map mfa --states new_local_states.npy --output new_scores.parquet
```

接口不接受ground truth；未知state映射到训练词表保留的UNK。不能传另一套重新编号的簇或全局matched ID。输出是冻结的异常/一致性分数，不是校准后的正确率。

可用 `--methods gru causal_transformer` 只加载指定方法，`--threads 2` 限制CPU线程。神经网络评分需要PyTorch（项目的 `gpu` extra，CPU上也可以运行）；Lambda环境已具备，本地只看报告不需要安装Torch。

## 参考

- [Hidden Markov Anomaly Detection](https://proceedings.mlr.press/v37/goernitz15.html)：序列异常检测背景；本轮 HMM 为直接生成模型，并非该文单类SVM目标的复现。
- [Deep One-Class Classification](https://proceedings.mlr.press/v80/ruff18a.html)：固定中心、约束编码器的依据。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)：VAE训练原理。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：注意力模型与因果掩码。
- [Scikit-learn novelty/outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)：经典检测器的新样本评分接口。
- [TabADM](https://arxiv.org/abs/2307.12336)、[NeuTraL AD](https://arxiv.org/abs/2103.16440) 已调研；其特定损失和样本拒绝/变换方案未在本轮复现，不能列为完成项目。
