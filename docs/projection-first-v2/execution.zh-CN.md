# 局部投影优先 v2：执行协议

采用用户压缩包中的最终v2，原文件完整保留在request/。采用时间2026-09-24 07:40:59 UTC；交付目标为次日同一时间，H20后不启动新批次，至少保留最后两小时用于审计/报告。原来已冻结的256题×6条件继续完整执行，输入、代码和统计方案不变。

## 固定优先级与数量

1. 等原MATH主队列完成生成、原始审计、统计与报告；只启动一个CPU等待进程，不加载第二个GPU模型。
2. 新GSM8K官方train中64题×4条件：baseline、原local8、same-anchor shared8、same-anchor wrong-local8。
3. 原64道MATH验证题按固定哈希选择32题，运行D01–D10。优先D01/D02/D03/D05，其后D06/D04/D07/D08/D09/D10。
4. 原题目、全部回答、配对统计、几何指标、全部修复/损伤的盲核验页面与论文表图归档。

标准档新增576次，D07/D08各额外32次相同介入边界的baseline，使用原方案64次预留，故实际上限640次。不得把prefix16重放baseline无条件当成prefix8/32数值路径的对照。缩减档32道GSM×3条件及4项消融，共224次。档位在任何新outcome可见之前，按原smoke的p90耗时与剩余预算确定；单条件预算至少25秒。每批完整准入，不基于部分正确率停跑或扩展。旧256题不缩减。

## 与当前实现的准确对应

- 模型Qwen/Qwen2-7B-Instruct，revision f2826a00ceef68f0f2b946d945ecc0477ce4450c；bf16/SDPA，greedy，repetition_penalty=1，总生成上限2048。
- 第14个block对应model.model.layers[13]的输出hook。原方法自然生成16token，在同一自然前缀重新构建完整cache，仅修改最后4个回答位置，然后移除hook自由续写。
- 使用revision-locality-20260923/decoders/p16_l14_tokens/decoder.npz，64个GMM锚点。分配为欧氏最近中心；local_basis是相对于该锚点训练的残差SVD基。不是经验均值锚点，不是MFA因子载荷。
- 原shared8保持同一锚点。原随机条件是同一种径向/能量匹配操作的三个seed42/137/271，按题内平均。
- 现有decoder有local_basis[64,32,3584]及shared_basis[512,3584]，故local16/shared64直接截取同一产物，无新增拟合。
- wrong-local8使用固定seed9242026的无固定点循环置换；只换基，保留recipient锚点。映射在新outcome之前写入plan.json。
- D09近零阈值1e-12，遇到近零重构则原样保留并计数。D10从窗口未修改快照计算平均改变量并一次性广播。
- D07/D08依然使用原prefix16训练的地图和基，属于冻结地图的介入时机探索，不宣称在各时机重新训练了相容地图。
- 新GSM题排除全部历史MATH5000、GSM官方test1319、原GSM train64确认题及文本重复，保留数据revision、样本ID、文本哈希与排除来源。这里只保证项目内未使用，不声称排除模型预训练接触。

## 校验与统计

本地/远端合约测试包括所有算子的独立逐行复算、原C1的逐元素一致性、正反符号、共同偏移保留token间差异、范数恢复与近零回退、预算档位，以及合成原始记录→配对统计→独立审计→HTML/PDF与篡改拒绝。随机小Qwen的真实CPU前向覆盖10个变体，核对no-op生成、修改后cache与完整重放、hook移除；这不构成科学收益。

原MATH确认统计原样导入。v2采用问题级2000次配对bootstrap，逐项95%区间；不宣称多比较后总体显著。随机三个seed不当作三个独立问题。所有消融标探索性，不替换主方法，不为赢家增加确认。几何指标只描述，不能在测试上拟合新gate。

每批独立重新检查token、自然前缀、模板、来源SHA、评分、实际bf16改变量、分区、短回答和完整配对。新增问题提前结束时不干预，保留分母。原始失败/未完成与模型答错分开；缺资产/预算未启动均保留记录。

程序输出所有修复/损伤的盲核验页面；自动格式、截断、答案变化标记不是语义审查结论。真正读过回答之后才能填写semantic review，不把页面生成冒充人工核验。最终须看PNG与渲染PDF并检查逐题链接及HTTP。

## 代码与结果

配置：configs/projection_first_v2_20260924.json。
准备：scripts/prepare_projection_v2.py（OpenAct环境，CPU）；执行scripts/evaluate_projection_v2.py；原始审计scripts/audit_projection_v2.py；统计和报告各自独立。单一监督scripts/run_projection_v2_queue.py等待原主队列。

新结果根目录：/lambda/nfs/dami/hss/projection-first-v2-20260924。
原结果根目录：/lambda/nfs/dami/hss/revision-completion-20260924。
代码本地测试→GitHub→Lambda fetch/ff-only；运行前冻结源码与输入SHA，开始后不改被冻结文件。必要恢复使用新命名空间并保留失败证据。实际运行状态见各根目录queue_status.json；本文件不是已得到实验结果的证明。
