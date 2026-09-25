# HSS 推广实验：执行审查 v1

用户规格日期 2026-09-25；探索性延伸，沿用已经检查过的问题。每项实验在计算干预结果前分别保存 plan.json（UTC、Git、配置、代码和输入 SHA256）。原始 v3 不修改。

## 全部固定

- Qwen2-7B-Instruct，模型 revision、BF16 SDPA、精确历史 token、EOS、因果右移评分均沿用 v3。
- hidden-state index L = model.layers[L-1] 输出。只干预生成位置；prompt 不动，因此首个生成 token 的预测不受干预。
- teacher forcing 每个条件使用相同的完整历史输入；后半段指最接近 core token 一半的句首，并列取较早。无内部句首则 half_token；保留终止 EOS 的 NLL。
- 复用 v3 MatchedTransform，包括 BF16 修正和所有审核门槛。失败则保留记录并停止，不放宽门槛或删除失败题。
- 100,000 次以问题为单位的 bootstrap，seed=20260925；所有新层和状态全部报告。记录原始逐 token logprob 和干预能量。
- 以 2026-09-26 01:59 UTC 为用户指定投稿结果冻结时间。完成和审计晚于此时标为讨论期；此处不声称已核实会议官方日期。

## 实验 1：题型、难度匹配

- 原 v3 的 96 个 sink B；候选是完整 N_B，不要求正确或 boxed，且与方向构造 A 完全不重叠。
- 匹配无放回，避免同一个对照被当作多个独立问题。seed 固定打乱 sink 顺序；先为全部题做同题型同 level 匹配，再为剩余题选同题型 level 差 1；无候选跳过并列出。优先完成精确匹配，避免放宽匹配抢占精确候选。
- 两组均采用 half_sentence（关闭 first_repeat），全生成位置干预，后半段评分。每题 NONE、C1；原 axis/tau 不改。
- 主终点：匹配对 (C1-NONE)_sink − (C1-NONE)_matched，95% 配对 bootstrap。
- 几何同时报告 z>tau 比例、平均正超出量、实际更新范数、长度、自动正确性；精确匹配子集为预先指定敏感性分析，不替代主结果。
- 解释：可检验结果是否超出已匹配的题型/难度，不能排除所有题目因素，也不能据此单独证明生成模式的因果机制。

## 实验 2：深度

- 新层固定 4,8,12,16,20,24；14 引用已完成 v3，不混入六重校正。
- 各层方向使用同一 S_A/N_A 的 response mean。阈值复用原 300 个 tau_ids 及原 vector_samples 中的 token indices，以各层未干预的 teacher-forced 激活估计75分位。
- S_B96 / M_B100、评分窗口沿用 v3。每层 NONE,C1,5ISO,5MAN。
- ISO seed+1；MAN seed+2，复用原 manifold_controls 规则。排除原 sink global ID 在该层对应的局部簇；若该 ID 尚未出现，使用全部中心并明确记录，不重新根据结果寻找该层 sink。
- 10 个方向对本层轴正交化。若有效中心对不足或正交退化则停止该项并报告，不替换层或挑方向。
- 每层 C1−对照均值，置信水平精确为 1−0.05/6；normal 为描述性95%区间。报告轴与14余弦、触及比例、能量和全部对照。
- 最大效应层不解释为独立于剂量的“最重要层”：各层自然截断的改动大小不同。

## 实验 3：其他状态

- index14 训练人数最多的4个非 sink 局部簇，人数并列按 cluster ID；不看干预结果选状态。
- 对每状态 in/out 各按题型半分（原 stratified_half_split seed）。out 排除该状态及 sink。
- 轴为 in-A均值减out-A均值；tau 取随机最多300条out-A、每条最多200token的75分位。人数不足照实记录，不补重复样本。
- B 两组各最多100条，固定随机无放回。half_sentence统一评分；每组 NONE,C1,10正交能量匹配方向；MAN排除该状态及sink。
- 主终点 in-B 的C1−对照均值；四重校正98.75%。out-B同指标、in/out差为次要95%区间；in/out独立重采样，不伪称题目配对。
- 状态之间允许人群交叉，记录ID；Bonferroni不要求状态间独立。报告训练占用、主导题型和正确率。
- 仅4个高占用状态，不外推为所有HSS状态；in组显著但in−out不显著时，不称状态特异。

## 实验 4：自由生成

- sink历史测试55题 + S_B97题，去重并核实共152；正常历史测试固定随机50题。记录来源分层。
- zero/C1/ORTH_MAN_1，index14，greedy，repetition_penalty=1，2048上限，prompt不动，fresh KV cache。沿用v3 axis/tau。
- 对照每步在本组当前激活上求C1目标更新范数，再用v3实现沿固定ORTH_MAN_1更新。在线条件只在同一个当前激活处匹配；各组生成后分叉，不声称跨组路径的总能量匹配。
- 正常和sink合计可复用的105条zero中固定抽10条重跑；token完全相等才复用全部105。若任何不一致，全部202题重跑zero；检查结果保存。
- 主终点 complete nonempty boxed AND token数<2048；sink人群两次双侧精确McNemar（C1 vs zero、C1 vs control），Holm校正。要求C1方向为正且两个校正p<.05才称行为方向特异改善。
- 自动正确性、完整boxed、长度、正常副作用、重新编码后sink退出/进入和净数全部报告。使用无干预模型重编码每组的实际生成文本，保持同一冻结response map。
- 不根据中途结果停在显著点。冻结前未完整跑完/审核则不进入投稿结果；所有部分结果保留。

## 实验 5

按用户要求留到讨论期，本轮不占用GPU。届时另行核实 Llama-3-8B-Instruct 历史解码与地图，冻结最低训练正确率且>=40题的中层状态选择；低正确率本身不等价于退化sink，必须报告行为构成。
