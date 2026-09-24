# Qwen–MATH token-mean：3%、4%、5% ICL 容差

用户要求在原来的 0–2% 之外，分别保存 3%、4%、5% 的 GMM 拟合结果。
复用 `qwen-math-ablation-20260918` 中每层 K=2…80 的已拟合对角 GMM，
不重新运行 EM，也不更改旧结果。每档容差独立导出完整 HSS Result。

## 定义

- Qwen2-7B-Instruct × MATH 全部 5,000 题，每题全部生成 token 的 hidden mean。
- 原始 3,584 维；不进行额外归一化、标准化、白化或 PCA。
- 主结果是 HF 位置 0–28（位置 0 为 embedding，位置 28 已过 final RMSNorm）。
- 另存位置 28 在 final RMSNorm 之前的单层对照 `pre_final`。
- 仅接纳收敛且 ICL 有限的候选。阈值为 `min_ICL + tolerance * max(abs(min_ICL), 1)`，
  在阈值内取最小 K。这是模型简化容差，不是 EM 的收敛 tol 或统计置信区间。
- 聚类、容差选择不使用正确性标签；5,000 题全部用于拟合，属于描述性分析。
- 局部簇号用 posterior MAP；跨层沿用 cosine Hungarian、eta=0.6。
  另外保存 nearest-centroid 局部归属，不能把它与 posterior 主结果混用。

## 执行

代码通过 GitHub push → Lambda fetch/fast-forward 部署。在 HSS 环境运行：

```bash
.venv/bin/python scripts/export_gmm_tolerances.py \
  --source /lambda/nfs/dami/hss/qwen-math-ablation-20260918 \
  --output /lambda/nfs/dami/hss/qwen-math-gmm-tolerance-20260924 \
  --tolerances 0.03 0.04 0.05
```

该命令只使用两个 CPU 线程；不存在 GPU 拟合路径。它读取已有 mean 缓存，
逐个重放选中模型在全部 5,000 行的 posterior 和 nearest 归属，要求与原缓存完全相同；
同时检查概率归一化、导出再加载后的参数逐元素相同，以及 Result 全文件校验和。
读取缓存快照必须与原快照及行顺序一致。缺失 K 扫描或未通过核验时不交付完成结果。

## 保存与加载

```text
qwen-math-gmm-tolerance-20260924/
  protocol.json
  selected_layers.csv
  cluster_counts.csv
  delivery.json
  icl_03pct/{post,pre_final}/
  icl_04pct/{post,pre_final}/
  icl_05pct/{post,pre_final}/
```

各 Result 内保存每层模型中心、对角方差、混合权重、完整 K 候选记录、
源模型 SHA256、posterior 概率、posterior/nearest 两种局部归属、跨层全局 state ID、
题目元数据/标签和描述性簇统计。没有符号链接，不依赖原拟合缓存加载模型。
`delivery.json` 仅在六份 Result 均通过审计后生成。

```python
from hss.results import Result
from hss.results.models import load_layer

root = '/lambda/nfs/dami/hss/qwen-math-gmm-tolerance-20260924/icl_03pct/post'
result = Result(root)
assert result.validate(full=True)['valid']
model, projection, selection = load_layer(root, 14)
# X: 同一定义、同一层的原始 token-mean，shape [n, 3584]
labels = model.predict(projection.transform(X))
```
