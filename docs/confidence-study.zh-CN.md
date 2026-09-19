# 输出置信度与正确性方向：后续检验协议

这是在已经查看过 channel-study 验证结果后的**探索性后续检验**。沿用原始 40/60 划分及 MMLU 去重；新结果不能声称来自全新、未打开的确认集。统计单位为题目。

## 固定定义

- 首 token 的 logits 使用 **prompt-last post-RMS** 状态和采集所用、固定 revision 的原生 LM head。t1 状态预测第二个生成 token。保存两者，分别分析。
- 完整词表熵（nats）、top1-top2 logit margin、概率 margin、top1 概率；不使用生成过程平均熵充当首 token 熵。
- CPU float32 重建；检查重建 argmax 与实际首 token，报告 BF16 舍入的影响，不以未经验证的完全等同为前提。
- 精度审计额外将 head logits 四舍五入回 BF16，并应用实际继承的 repetition penalty。Qwen2 checkpoint 默认 penalty=1.05；本轮报告原生分布与该生成策略的敏感性，不修改在跑采集。Teacher-forced 与 generate prefill 的数值差异仍可能导致不一致，原始 generate 首步 logits 没有被保存。
- 新主分析不做 medium-magnitude 筛选。旧 K16 仅作为固定的历史比较器，坐标来自旧实验 discovery 排序。
- 预生成 nuisance：题型、难度、log(prompt 长度)、log(RMS)。诊断 nuisance 另外加入首 token 身份（未来变量，不称作预生成预测器）。
- 所有标准化和参数拟合只用 discovery。L2 logistic 的 C 在 discovery 内四折选择，网格 1e-4 到 1；重新在全部 discovery 拟合。验证集不参与挑选。
- 标量同时报告“数值越大”的原始 AUROC 和 discovery 决定符号后的 AUROC；跨数据集冻结源 MATH 的符号，绝不根据 MMLU 标签翻转。
- 增量比较：nuisance → +熵 → +固定 K16；补充 +所有置信度标量 → +K16 / +全维残差。配对题目 bootstrap 1,000 次，95% CI。实际等价界限预先定为 ±0.01 AUROC；CI 跨 0 不自动判等价。多个描述性比较不作家族显著性主张。

## terminal-readout 的定义核对

参考仓库 `xiangli-damien/terminal-readout` 的实际工作分支 `codex/pilot-infrastructure`，commit `0ac9b89ee6318cfbd3a1107f76c5a3f68a5d71ca`，`docs/readout_geometry_zh.md`、`src/terminal_readout/readout_geometry.py` 和 amendment008。最初只读到旧 main `092aca6`，随后通过用户指出的本地 Codex 项目找到已完成的几何实验，纠正了“没有定义 v”的初始判断。两者记录均保留；计算公式没有因此变化。

原项目在 Qwen2.5-0.5B 上发现单一最弱读出方向，并记为 v；尚未证明它有通用置信度功能。本轮在各个新模型自己的 head 中重算相同定义的方向，不能将 896 维原模型向量原样搬到 2048/3584 维新模型。其引用的 [Confidence Regulation Neurons](https://arxiv.org/html/2406.16254) 讨论低奇异值子空间。

本轮 v 是**同定义、各模型重算的对应方向**；其“置信度方向”功能仍是候选解释。设原生 head 为 W、RMSNorm 权重为 gamma；softmax 去掉公共 logit 后的 pre-RMS 有效线性矩阵为 A=(W-mean_vocab(W)) diag(gamma)。

- `v_min`：A 最小奇异值的单位右奇异向量；最大绝对坐标取正确定符号，完全不使用题目标签。报告底部谱与 eigengap。
- `low_readout_fraction`：pre-RMS 状态落在底部 floor(0.01 D) 个右奇异向量的范数比例。补充 signed projection、绝对 projection 和低子空间能量。该子空间不自动等于置信度。
- 全维正确性 w 在 pre-RMS 坐标比较；standardized probe 系数必须除以 discovery 标准差后才能与 v 求余弦。另做 uniform/global scaling 的 L2 拟合，检查结果对坐标标准化的敏感性。
- 难度方向：discovery 上 ridge 回归 MATH level；题型是多类，使用 ridge 系数张成的子空间，报告 w 的投影范数比例，而非虚构一个题型向量。另记录从 discovery 熵拟合的方向，区别于权重定义的 v。

## 时间与层

复用所有层 prompt-last，另外只提取生成 token16 的所有层。固定 prompt-last/t1 的全维方向跨位置、跨层读取；同层 discovery 重新拟合的线性 probe 为辅助诊断，用于区分“冻结方向失效”和“线性可读信息不存在”。最终层使用 pre-RMS，与其他 raw residual 层一致。同一图使用相同 n_tokens>=16 的题目集合。

因果 decoder 下，prompt 位置的状态不会因后续生成改变。因此生成后仍能读 prompt 状态，不是“模型确实通过 attention 回取”的证据；需要干预/KV 路径实验。

## 条件性因果实验

本轮先执行已有数据的观察性检验。只有明确候选、确认接口并与在跑采集错峰后，才能执行上游 residual 干预和采样实验。当前状态必须报告 `not_run`。

任意方向的 pre-RMS 改动一般能改变 argmax。只有**精确** softmax-null 方向使 A v=0 时，改动该方向仅通过 RMS 分母给 centered logits 正比例缩放；低奇异值方向只近似满足条件，不能保证 greedy 不变。

“熵之外增益”仅排除所测标量/模型充分性，不证明模型知道的超过完整输出分布。方向与难度对齐也不证明因果机制。

## 运行

额外依赖为 PyTorch 和 safetensors（只用于在 CPU 读取权重）；可用 `uv sync --extra readout` 安装。本轮没有加载 decoder 到 GPU。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/study_confidence.py --stage scalars
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/study_confidence.py --stage analyse
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/study_confidence.py --stage precision
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/study_confidence.py --stage layers
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/study_confidence.py --stage layer-analysis
.venv/bin/python scripts/study_confidence.py --stage report
```

可选的 `--stage readout-check` 是256个发现题的直接读出／匹配温度检查，不能替代上游 block 干预或采样正确率。`--stage audit` 独立核对样本身份、保存系数、AUROC 与方向能量。
