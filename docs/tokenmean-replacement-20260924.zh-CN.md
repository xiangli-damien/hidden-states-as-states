# 冻结 token-mean 地图的功能替换对照

用户于2026-09-24要求用之前的 token-mean cluster 试重构。新结果另存
`tokenmean-replacement-20260924`，不改任何已封存实验。

## 冻结协议（查看新干预结果之前）

- Qwen2-7B-Instruct，同一revision，block14，前16个生成位置，bf16/SDPA。
- 原全回答mean GMM：5,000道MATH、raw3584维、2% ICL容差、K33，冻结已有中心/方差/权重，无重新聚类。
- 按原posterior MAP分配3011道训练题的全回答均值；仅用这些均值拟合每簇rank8残差SVD和共享rank8。固定GMM中心，不增加经验中心，不使用正确性标签。记录每簇样本数和实际可用rank。
- 干预时只用已生成16token的均值选一个区域，绝不使用未来完整回答的均值。全回答mean到前缀mean存在分布偏移，报告后验、占簇数、log-density及MATH原全回答簇的分配一致率。
- **mean_shift**：重构均值m得到m_hat；每个token加上同一m_hat-m。保持token之间差异，仍保留高维逐token残差，不能称每token8维压缩。
- **token_project**：前缀均值选同一个区域，在该区域对每个token各自做中心加8维投影。这里每token8坐标，但方向是从全回答均值学到的，并非重新用token拟合。
- 每种算子都有centroid、shared8、local8、三个固定无固定点错基置换。另运行identity及原token地图centroid/shared8/local8桥接，共16条件。
- 复用上次历史MATH测试32和GSM功能64题、完全相同前缀及原参考续写。不是新增独立确认。原mean地图看过全部MATH，MATH明确标transductive；GSM不参与拟合，但已被此前研究使用。
- 主比较GSM mean_shift local8−shared8，KL(p_original || p_patch)及完整原模型参考续写NLL，两者均报。按问题2000次配对bootstrap、逐项95%区间，无多比较校正。错基先题内平均。
- token_project与token地图对照为次要；K33/64、训练表示、分配规则不同，不把差异全部归因于mean操作。这里不做自由生成或正确率主张。

## 执行和审计

先CPU拟合和算子测试，再真实模型2题（MATH/GSM各一）全部条件smoke，随后同一有限96题。每条件新cache、只patch一次，实际激活与旧capture逐元素相同；保存理想/实际bf16 patch、next-token logprobs、逐参考token损失及原文。独立审计另行复算posterior、投影、mean_shift不变量、bf16量化、KL/NLL、身份和完整条件覆盖。原token地图桥接核对已有logprobs/损失。之后生成配对统计、图和逐题页面。

单GPU；CPU/BLAS最多4线程；不升级环境。代码本地测试→GitHub→Lambda fetch/ff-only。状态见新根目录queue_status.json；本协议不代表已完成运行。
