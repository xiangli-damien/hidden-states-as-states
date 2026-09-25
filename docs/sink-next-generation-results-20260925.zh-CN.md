# 在线 C1 截断：完整生成结果

生成606/606；服务器审核于2026-09-25 22:13:31 UTC通过，本地独立核验于22:14:07 UTC通过。均早于用户指定冻结时间。202道题，三组；全部结果保留。

## 结论

**C1提高了历史候选题的完整boxed完成率，相对不干预的改善通过Holm校正；但没有显著超过正交对照，因此预注册的“两组都胜过”标准未通过。自动正确率没有可靠提高。**

这给出一个有限的行为收益：更多回答以完整boxed形式结束。不能写成方向特异的纠错成功、所有回答更好，或推理状态得到因果确认。

## 设定与主终点

- Qwen2-7B-Instruct；hidden-state index14，即`model.layers[13]`输出；只在转发生成token时在线截断，prompt prefill不改，首个生成token不受干预。
- 使用原2% ICL容差、K33回答均值地图的local8候选区域及已冻结轴/阈值。近期0%容差global58的描述性分析没有用于本次干预。
- 152道历史候选题（55 historical_test＋97 S_B）、50道正常对照题。正常题不是“本次baseline必须答对”的筛选。
- 三组：zero、C1、ORTH_MAN_1。主终点为存在完整非空`boxed`且新token数小于2048。原自动正确性为次终点。
- 两个预定比较：C1对zero、C1对ORTH_MAN_1，双侧精确McNemar，Holm校正。95%配对bootstrap区间逐项报告。

## 候选题：152题

| 指标 | zero | C1 | ORTH_MAN_1 |
|---|---:|---:|---:|
| 完整boxed且未到上限 | 72（47.37%） | 90（59.21%） | 81（53.29%） |
| 冻结自动正确 | 19（12.50%） | 21（13.82%） | 15（9.87%） |
| 平均新token数 | 1022.49 | 1082.74 | 1074.23 |
| 达到2048上限 | 17 | 21 | 19 |
| 无干预重编码后仍属于候选区域 | 76 | 34 | 70 |

| 主比较 | 主终点改善/损伤 | 净变化 | 95%配对区间 | 双侧精确p | Holm p |
|---|---:|---:|---|---:|---:|
| C1−zero | 36/18 | +18题；+11.842个百分点 | [2.632,21.053]个百分点 | 0.0198343 | **0.0396687** |
| C1−ORTH_MAN_1 | 31/22 | +9题；+5.921个百分点 | [−3.289,15.132]个百分点 | 0.271679 | **0.271679** |

自动正确性相对zero：11题错→对、9题对→错，净+2题；差值+1.316个百分点，独立精确配对bootstrap区间[−4.605,+7.237]个百分点，未校正双侧p=0.823803。这不是可靠的纠错收益。

## 正常对照题：50题

| 指标 | zero | C1 | ORTH_MAN_1 |
|---|---:|---:|---:|
| 完整boxed且未到上限 | 36 | 35 | 39 |
| 冻结自动正确 | 23 | 21 | 21 |
| 平均新token数 | 537.94 | 501.44 | 528.64 |
| 达到2048上限 | 3 | 2 | 2 |

C1相对zero：主终点新增3题、损失4题；自动正确性修复2题、损伤4题。这个人群没有观察到净收益。

## 区域退出是另一个指标

152道历史候选题的新zero中，76条在当前候选区域，76条不在。C1让前者中的50条离开，同时使后者中的8条进入，净退出42；对照退出23、进入17，净退出6。

这些是预定次要描述。新文本在**无干预模型**中重新前向后分配区域，不是被干预时的在线状态。离开该几何区域不等于答对；应同时报告19→21的自动正确数。这里没有以事后选择的区域退出终点替代未通过的主判据。

## 方法边界与语义检查

1. 这是沿用此前已检查问题的探索性扩展；无标签地图拟合曾包含全部5000题，不能称独立新数据确认。
2. 在线对照在自己的当前激活上匹配C1规则；文本分叉后累计能量不同。候选题的每题平均总更新能量为C1 10974.90、对照24225.05，不能称跨组整段能量相等。
3. 历史采集与这次新生成的设置差异已在原协议披露；历史区域成员不要求在新zero里仍属于该区域。
4. 原自动评分保留，未因为某一口径更好而替换。已有43题的AI辅助、非盲全文检查发现解析误差及“最终数值对但推导错”；其余159题仍待语义复核。完整生成/统计完成，不等于202题数学推导已全部审核。

## English results paragraph

In an exploratory evaluation on 152 questions whose historical responses occupied the candidate region, online clamping at hidden-state index 14 increased complete, nonempty boxed answers within the 2,048-token budget from 72/152 (47.4%) to 90/152 (59.2%). The paired increase was 11.84 percentage points (95% bootstrap CI [2.63, 21.05]; Holm-adjusted exact McNemar p=0.0397). The orthogonal control achieved 81/152 (53.3%), and the clamp did not significantly outperform it (difference 5.92 points, 95% CI [−3.29, 15.13]; adjusted p=0.272). Thus, the preregistered requirement to improve over both comparators was not met. Frozen automatic correctness changed from 19/152 to 21/152, without a reliable improvement; on 50 normal comparison questions it changed from 23/50 to 21/50. The result supports a limited improvement in answer completion over no intervention, while behavioral direction specificity and accuracy benefits remain unestablished.

## 审核与文件

- [完整152＋50题三组原文与统计](http://127.0.0.1:8794/report/steering_final.html)
- [实验1–4合并报告](sink-next-results-20260925.zh-CN.md)
- [完整冻结统计](sink-next-20260925/exp4/summary.json)；[独立本地复算](sink-next-20260925/exp4/independent_local_audit.json)
- 原始结果：`/lambda/nfs/dami/hss/sink-next-20260925/exp4-v3`；本地镜像：`results/sink-next-20260925/exp4-v3`。
- 独立审核重新核对606个JSON/NPZ哈希、202题配对、主终点boxed解析、378959个在线更新步骤；实际相对能量误差最大0.00169412，低于冻结0.002门槛。用组合数计算双侧精确检验、用卷积计算配对bootstrap精确分布，与冻结审核一致。服务器另已重放冻结GMM分配。
- Plan SHA256：`e97aab910a200e18c589f63ef3ea5730a5ffe680ddd18ff19f4cad16c661a51d`。
- Summary SHA256：`ca9d19d7a5ffd5822a4c4bd3821f108b5a3e8e5862f057b5b736176014b30ef7`。

不启动新的科学实验，不改变已冻结的地图、轴、阈值或解码设置。
