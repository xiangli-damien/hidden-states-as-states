# HSS revision：实验执行清单（2026-09-23）

## 目标和证据边界

本计划把用户提供的完整讨论（E00–E26）转成可复核的执行清单。目标是检验 region/state 保留什么信息、支持什么干预，以及跨层关系是否提供独立预测。**不承诺正结果，不把代码存在、smoke 通过或作业启动写成实验完成。** 阴性结果、失败的干预和未收敛拟合均保留。

参考：[From Directions to Regions](https://arxiv.org/abs/2602.02464)，尤其 §3、§6、附录 D/E。其 MFA 使用 region、局部连续坐标及 responsibilities；因果定位另有监督 DBM；主题 steering 不能直接等同数学纠错。本文采用这些已有工具，不把采用 MFA、几何重构改善或普通 steering 本身作为新增方法贡献。现有 HSS MFA 是 component-specific diagonal noise 的 EM 实现，与该文 shared-noise/gradient training 不同，不能标作其逐项复现。

## 统一协议

- 主模型 Qwen/Qwen2-7B-Instruct，revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`；确认模型 Llama-3.2-1B-Instruct，待主要实验稳定后复制。
- 真实任务：已采集 MATH 5,000；新增 GSM8K main/test 全量 1,319。GSM8K pin `740312add88f781978c0658806c59bc2815b9866`。使用与 MATH 相同的 zero-shot CoT/boxed 指令，单独记录模板，避免把输出格式变化混同数据集迁移。
- MATH 复用 question-level 3,011/1,003/986 划分。旧测试集已被探索过，本轮 MATH 结果标作探索性复核。所有选择仅用训练/验证；GSM8K 的新确认集用于冻结后的迁移评价。
- GSM8K 以 ID 哈希固定 adaptation/validation/confirmation 划分（40/20/40），不依据标签分组。严格冻结 map+readout、冻结 map 后重新校准、目标域重新拟合三个设置分别报告，使用相同 confirmation 集。
- **几何数据不 normalize。** 任何 PCA 中心化、readout 标准化或 norm-matched 对照单独记录，不能悄悄替换 clustering 输入。
- patch 对象为 1-based decoder block 输出；最后 block 使用 final RMSNorm **之前**的 residual。post-norm 仅作为独立观察对象。prompt-last 只是最小范围对照，同时测试连续 4/16 个真实 token；分别考察题目内容末尾、完整 chat prompt 末尾，以及已生成 16/64 token 的前缀末尾。每个时点只使用当时可见的 prefix。
- 为解决 bf16 full-sequence 与 prefill shape 差异，干预 codebook 使用重新提取的 **prompt-only prefill** 激活。不能直接拿 generation-mean 的中心 patch 真实 token；旧 full-sequence prompt-last 仅作形状差异审计。
- 每次单层单位置干预：重新 prefill、重新构造受影响的 KV cache；只 patch 一次。identity hook、hook 调用次数、实际替换位置、baseline/identity logits 和 continuation 一致性必须通过。
- teacher-forced KL/NLL 比较相同 prefix/同一已保存参考续写；自由生成另测答案一致性与真实正确率，不能混用。
- 源代码必须本地 commit → GitHub push → Lambda fetch/ff-only。原始数据与旧实验不覆盖。
- 统计单位是问题；bootstrap、划分和 donor 分组均不能把同题 token 当独立样本。区间明确 pointwise / 多重比较范围。pilot 不标作确认结果。

## 今晚先执行的顺序

1. E00：冻结协议；检查数据/标签与模型版本；新 prefill 提取及 identity-hook 测试。
2. E03/E16：CPU 上拟合 prompt-last 的 GMM/KMeans 和重构 baselines，同时检查 held-out alignment。GMM 主分配 nearest centroid，posterior 分配作为单独消融。
3. E04：代表层的同-prefix KL、完整参考续写 NLL、zero/mean ablation、residual 恢复曲线；独立的自由生成行为 pilot。
4. E21 数据前置：GSM8K 全量 HF greedy 采集、原始标签评估及 OpenAct 验证，保存全生成激活、prompt-last、mean 与末层 norm 两侧。按小分片落 SSD，SHA 校验后发布 dami。
5. E08/E12：构建中间变量 × 表面形式 × 非目标变量交叉的算术任务；先测原模型能力与完整 donor patch，再比较 state-only / translation / 匹配对照。
6. E14/E15：冻结真实任务 steering 操作与验证网格；跑净正确率/损伤 pilot，再据实际效应决定正式样本量。
7. E21：GSM8K 数据齐全且源域方法冻结后，运行三个明确区分的迁移设置。
8. E07/E09/E11/E17/E18：在前置产物齐全后补公平比较、短程转移预测及 anchor-layer 干预迁移。

CPU 与 GPU 可以同时做不同阶段；GPU 上的 decoder 作业排队，避免争抢显存。每个阶段必须有 `status`、逐样本原始结果、配置/代码/输入哈希、完成审计，失败时保留检查点。

## 全部实验清单

“风险”是严格对照下得不到预期正结论的风险，不是实现完成概率。状态在运行清单中更新；此表不代表这些实验已经实现。

| ID | 优先级 / 风险 | 必须回答的问题与完整对照 | 交付与判定条件 |
|---|---|---|---|
| E00 | P0 / R1 | 问题级 split、重复问题、标签、模板、EOS/padding/空样本、block/norm 位置、prefill/cache、identity hook | 输入清单、固定配置、identity 与原模型一致；不通过则阻止因果阶段 |
| E01 | P0 / R2–3 | 同数据/split/assignment 的 GMM、MFA、KMeans、global PCA/FA；同 K 与参数/表示预算两类对照 | 几何、行为、成本分别报告；nearest 与 posterior 不混合 |
| E02 | P0 / R2–4 | MFA K×rank×样本量、seed、噪声结构；训练收敛和占用 | held-out LL/NMSE、有效占用、uncertainty、partition stability；边界最优明确标记 |
| E03 | P0 / R2 | state-only centroid、hard/soft MFA、同 K KMeans、global PCA、local PCA、共享 PCA、大小匹配随机分组 | NMSE 分母固定为训练均值；EV、cosine、norm error；ID/连续坐标/参数量明确 |
| E04 | P0 / R3–4 | 单层真实 token 的重构；原始、identity、zero、mean、centroid、local/global PCA；β=0,.25,.5,.75,1 | 同-prefix KL、参考 ΔNLL、loss recovered 的原始三项 loss、自由生成答案一致率和 accuracy |
| E05 | P2 / R4–5 | 单点通过后逐步增加 token/层替换范围 | 单点与累计替换分别报告，不能靠周围未改上下文宣称全模型压缩 |
| E06 | P1 / R3 | region、局部连续偏移、重构残差、原始状态的同预算 readout 与 β 恢复 | 有效信息的位置；不可把完整 residual 加回的 identity 称重构成功 |
| E07 | P0 / R3–4 | 题型/难度、prompt 与回答长度、首 token、norm、EOS、截断、格式、重复；排除答案/特殊 token 及固定 prefix | 控制前后效果；预生成不使用未来长度；sink 成因分开 |
| E08 | P1 / R2–4 | u=(a+b) mod10、y=(u+c) mod10；模板、u、c 交叉，额外非目标变量 | 未见模板/问题上的原任务准确率与无监督 state 的变量组织；不是 MIB/RAVEL 正式复现 |
| E09 | P0 / R3–4 | raw kNN、PCA、KMeans、linear/nonlinear probe、输入/logit；CoE/CLUE 必须遵守原定义和信息时点 | 同层信息、样本、FAR与成本；未实现方法不以简化代理冒充 |
| E10 | P2 / R3–4 | 固定表示比较 NB、Markov、one-hot LR、浅 MLP；有限网格 | 表示×readout 对照；验证无稳定改善即停止扩大搜索 |
| E11 | P0 / R3–4 | 在线 prefix monitor，验证集阈值，统一 non-final boundary 的 question-level FAR | detection/FAR、saved tokens/FAR、PR、校准、test FAR、误停代价与提取成本 |
| E12 | P1 / R3–4 | donor、centroid-only、centroid translation、local subspace；同/异 state、变量/模板/距离匹配 | 预定 counterfactual IIA、非目标信息保留、双向 donor、未见模板；full donor 不可操作先诊断位置 |
| E13 | P1 / R2–3 | 简单预定行为目标；插值/差向量/局部方向，扰动范数匹配 | 目标达成和任务保留；不冒称数学纠错或新的 MFA 算法 |
| E14 | P1 pilot / R4 | MATH 条件化 target；同类/难度且训练支持足够；自身中心、随机、KMeans、全局差均值、连续 probe | 验证选 layer/operator/α；冻结后测错→对、对→错、净正确率、格式/截断与覆盖率 |
| E15 | P1 pilot / R4 | 固定 action 的异质响应；state 对比 baseline risk、输入语义、连续状态与实际幅度 | 按问题划分的 held-out effect prediction；增量价值不成立即不称 functional regime |
| E16 | P0 / R3–4 | 训练上 cosine Hungarian / membership matching；测试真实流；随机、大小匹配与边际基线 | held-out matched flow、excess over marginal chance、split/merge；NB 重命名不变性只作实现检查 |
| E17 | P1 pilot / R3–5 | 一步/短程/干预后分开；state-only、局部坐标、连续低秩、边际/持久性 | next-state log loss、稀有转移、真正 rollout；不得每步读取真实后续状态 |
| E18 | P2早期pilot / R4–5 | anchor 层固定目标，经 geometry/membership/功能映射迁移；打乱跨层样本配对 | 不在每层重新挑最佳目标；跨层行为方向和选择性预测 |
| E19 | P2 / R4 | E07 后测试 sink 的可恢复窗口，匹配层/生成时间与干预预算 | 格式失败与推理失败分开；marker 不自动等于致因或 attractor |
| E20 | P2 / R4–5 | 只针对稳定的 E12/E18/E19 现象做 attention/MLP、破坏与恢复、路径对照 | 表示改变与行为效应的路径证据；前置未成立不扩大 circuit search |
| E21 | P1 / R3–4 | MATH→GSM8K：全冻结、仅校准、目标域重拟合；样本量匹配；MMLU 后续远域 | 距离/密度/支持覆盖、NMSE、AUROC/校准与置信区间；高posterior不等于in-distribution |
| E22 | P1复制 / R3–4 | Llama-3.2 复制首要 reconstruction 与干预结果 | 同定义复制核心终点；不是只添加 K 曲线 |
| E23 | P3 / R5 | 同受控变量下跨模型功能对应，禁止直接比较不同 hidden space 坐标 | 后续探索，不占首夜主要 GPU 预算 |
| E24 | P2 / R4 | 固定 actuator 比 HSS/probe/random/fixed trigger；再固定 trigger 比 actuator | 无未来信息、相同次数/预算、净收益与误停；依赖 E11/E14 |
| E25 | P2单步/P3多步 / R4–5 | 冻结响应模型前瞻选 action，执行一次；固定/global/continuous policy 对照 | 不在 test 遍历 action 后挑 oracle 当实际 policy |
| E26 | P2曲线/P3理论 / R3–5 | K/rank/softness/transition 容量—保真度与行为曲线 | 包括 decoder、codebook、连续信息和外部上下文成本；不宣称唯一最小 state 数 |

## 报告规则

每项报告包含：问题、实际配置与样本量、已执行条件及缺项、原始文件位置、主指标和配对区间、对照、失败案例、支持/不支持的主张、下一项的进入条件。结果只在审计完成后填入，表格不放示意数字。

## 前提树：先验证研究对象，再扩大干预（根据最新讨论修订）

**大前提 A：我们测到的是有用信息，而不只是题目语义、格式或长度。** 小前提 A1：prompt-last、题目末尾、生成前缀代表不同信息时点，不能互相替代。A2：在相同问题划分、相同 prefix 上比较 last-token、4/16-token mean、逐 token state 分布与顺序；完整回答均值只能标作事后诊断。A3：控制题型、难度、输入长度和当时的 next-token entropy；不能在 prompt 阶段用最终回答长度作可部署基线。A4：跨模板/跨题型/跨数据集检验与 IID 指标分开。

**大前提 B：token mean 是值得保留的压缩。** 它不是先验事实。均值不可逆地去掉顺序，并可能因方向抵消变小。比较同一窗口的 mean、最后位置、相同 token 数的无序 occupancy 和有序状态；另外报告 token norm、mean norm 和 coherence。mean 的有效性取决于重构/预测/迁移的实际结果。逐 token 表示拥有更多信息，必须同时报告存储预算；不能把其性能优势直接解释为方法更好。跨层 mean-delta 等于 mean 后跨层相减；同层相邻 token 的 signed delta 平均会望远镜式退化成首尾差，不能称保留完整动态。

**大前提 C：真实位置上的 state 重构保留功能。** 用实际 token 拟合 token codebook，并各自重构；不把 generation-mean 或 window-mean 中心广播到所有位置后称作正常重构。可以另测广播，但只能叫 stress control。先做逐位置 identity，再做 1/4/16-token 范围、相同累计扰动能量，以及随机位置对照。题目尾部与 chat 尾部单列，避免其实只动了 assistant header。一次单层窗口 patch 后重算全部下游计算和 KV cache。固定 prefix 的 KL/NLL 与重新自由生成分开。

**大前提 D：一个可定向的干预效应支持功能解释。** 扩大窗口能增大破坏，也可能降低选择性，因此不能把“变化更多”当证据。只有目标行为改变、非目标信息保留，且优于等能量随机和自身中心收缩对照，才进入 steering/反事实的正式检验。prompt 与生成中途分开；在相同可见前缀上从干预点继续生成，不能沿用干预前缓存。若完整 donor patch 都无法传递预定变量，先诊断变量分布和位置，不归咎于聚类。

**大前提 E：几何 state 在任务变化后仍有意义。** GSM8K 上先冻结 MATH codebook 和读出，再分开做校准与重拟合。coverage、重构、预测及行为迁移分别评价；“有簇可分配”不等于“是同一个功能 state”。

首轮实现使用第 7/14/21/28 block（1-based）、prompt / generation-prefix 16 / 64 三个时点，所有原始输入 token ID 沿用 OpenAct 保存值。短于指定生成前缀或含终止 token 的样本记为该时点不可用，不用末 token 填充。先完成前提 A–C 的可审计 pilot，再根据结果扩展；不是把所有条件的笛卡尔积一次跑完。

GMM 首轮是可运行的基础与对照，不意味着已经完成 MFA 比较。受控算术任务将明确标注自建 benchmark；没有跑官方代码/数据时不能写 MIB 或 RAVEL 复现。当前量级下的探索性正结果必须经过冻结后的新数据/第二模型确认，才升级为论文主要发现。
