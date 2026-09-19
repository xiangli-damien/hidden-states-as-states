# NDR / CoE 的 final pre-RMS 消融

输入为当前 OpenAct 的 response hidden mean。按稿件《From Complexity to Clarity: A Systematic Study of Geometric Dynamics in Hidden Representations of LLMs》公式 2、3、4、7，包含 embedding 和全部层；全响应已保存 token（含 EOS/特殊 token），不含 prompt。

只替换终端槽：原生 HF hidden_states 的 final post-RMS mean vs final_norm/pre/mean。中间槽不做归一化，也不做 PCA、中心化或白化。先对各 token 保存 pre/post 再求均值，绝不将 RMSNorm(mean(h)) 当作 mean(RMSNorm(h))。

- NDR = mean_layer(norm(mean_token(h))) / norm(final mean_token(h))。
- CoE-R 为平均归一化步长减去平均归一化转角。
- CoE-C 为归一化步长与原始转角映射到复平面后的均值模长。
- NDR、CoE-R、CoE-C 均保持论文“越高越正确”；末层范数以负号作为“正确更小”的单变量诊断。测试集上不翻转方向。
- 浮点运算用 float64。零范数/零端点位移/零端点夹角直接报错；当前数据不能静默插补。

每个 MATH 模型 5,000 条。Llama MMLU 原始 14,042 条，复用先前与标签无关的 prompt 去重规则，分析 13,937 个唯一问题。AUROC 复用先前冻结的 40/60 问题划分；分数不训练，范数差报告全体样本。该验证分区已在前序研究中使用，故本次是探索性复核。

AUROC 和 pre-minus-post 差的区间来自 1,000 次按正确性分层的成对 bootstrap。末层组均值差也用 1,000 次分层 bootstrap。组均值范数曲线阴影采用独立问题均值差的正态近似区间。Cohen d 为正确减错误的 pooled-SD 标准化差。统计单位是问题，不是 token。

敏感性分析：排除截断/解析失败；不含 embedding；在发现集确定的 10 个长度 quantile bins 内，只比较同 bin 正确/错误对，按可比较对数加权 AUROC。最后一项只有描述性点估计，bin 内仍可能有长度残差，不能声称完全排除长度影响。

最终 block 的真正残差更新是 pre[L] − hidden[L−1]；post[L] − hidden[L−1] 混入了 RMSNorm，不能直接归因于 block 的反对齐。报告同时输出 norm(pre[L])/norm(hidden[L−1])、相应更新 cosine，以及 post 端点的混合变化供比较。这里计算的是响应均值层面的变化，不等于每个 token 都有同样变化。

当前 Llama 为 Llama-3.2-1B，与稿件的 Llama-3-8B 不同。当前提示/生成协议沿用 OpenAct 的已采集版本，本研究旨在配对检验 pre/post 差异，不声称原论文每项数值完全复现。

## 复现

先准备 configs/channel_study.toml 对应的 summary cache 和原 study 的冻结分区，然后在 HSS 仓库中运行：

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/study_mean_geometry.py --config configs/mean_geometry.toml
```

输出：每题 samples.parquet、各层 norm_profiles.csv、完整 analysis.json、SVG/PNG 图片、index.html，以及固定源模型/采集 manifest、缓存和分区身份的 provenance.json。全程 CPU，不加载 decoder、不影响 GPU 采集。

