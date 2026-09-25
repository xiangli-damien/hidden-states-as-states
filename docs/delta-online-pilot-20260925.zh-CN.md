# 第一轮：逐 token 层间更新的在线风险与功能检查

用户已接受 Qwen2 × MATH、复用现有五个层对 GMM 的第一轮诊断。此前的静态状态 steering 已全部完成，没有可靠净纠错收益。此轮是新问题，不复用其正面验证集结果作为新结论，也不启动未冻结策略的自由生成搜索。

## 冻结范围

- 层对：0→1、6→7、13→14、20→21、27→28，末层必须 final RMSNorm 之前。
- 复用 local-depth-changes-20260921 的逐 token 对角 GMM；每层 K32、原 ICL 网格上界，不宣称全局最优。原始向量不做 normalization、PCA 或 whitening。
- 原 train3011 / validation1003 / test986，按题划分。训练地图用过训练题完整回答；测试时只取当前可见前缀。数据和自动答案标签已被反复探索，不是新确认集。
- 原始状态和更新量各自保留对应冻结 GMM，使用原 posterior MAP；不改成最近欧氏中心，不需要 Hungarian 编号匹配。

## A：在线风险的离线回放

在已观察16/32/64/128个生成 token 时，对仍在继续的回答预测最终失败。各时间点单独训练和校准，主比较为64。结束过早的题目不填充未来、不补零当正常路径；报告各阶段覆盖与样本数。不同时间点的风险集合不同，不能把AUROC曲线解读成对同一组问题必然逐步改善。

每题构造三种特征：可见前缀的逐层簇占比；加入同一个token在相邻选中层对之间的联合计数；再加入同一层的相邻token转移。所有表每题等权；层间词表独立。用训练标签学习Laplace α=1的类别条件复合贝叶斯分数，明确它不是独立观测的精确生成似然。固定种子的前缀内时间乱序只破坏时间顺序，不使用未来。

旧OpenAct只保存整段平均entropy，不能作在线控制。对保存的final_norm/post逐token和prompt-last状态，使用固定版本Qwen2的真实bf16 lm_head重新计算原始分布：当前熵／margin、已经观察的前p个token的平均熵和平均NLL。时间索引0是prompt-last预测回答token1；索引p预测下一token，绝不把该行的未来目标NLL加入控制。其余控制只有prompt长度和前缀token类型比例。bf16 batch shape可能带来小舍入差异，保存真实前缀前向与存储状态读出的KL检查。

训练五折OOF贝叶斯分数用于与控制变量组合；验证集选择逻辑回归C、概率校准和10%FAR阈值。冻结模型及分数后才计算测试指标。主比较：delta完整轨迹＋控制相对控制、相对同粒度state完整轨迹＋控制。其余时间点和占比／深度／时间边比较为探索。2,000次按题正误分层配对bootstrap，逐项区间、不含重训或多重比较校正。

输出是最终失败风险，不给某个token贴“正确／错误步骤”标签。当前最终层entropy基线要完成当前token前向才能取得，不能直接假装是中间层干预前可用的信号。后续若设计中层gate必须单独处理时序。

## B：本层更新的功能检查

在历史测试986题中用预先固定哈希选32题，不按label或新效果挑选。block14；观察16/64token后，修改最后1/4个生成位置；各条件比较相同原始前缀和接下来最多32个原始参考token。

八个条件：identity、zero_update、global_update、delta_centroid、delta_shared8、delta_local8、delta_wrong8、state_local8。更新方法加回本层真实输入。zero_update在输出处放回输入，内部attention缓存仍来自真实block计算；不把它称为物理跳过该block或省计算。state_local8直接重构完整状态，信息预算不同，只作描述对照。

GMM完全冻结；共享和局部rank8残差SVD只用原训练token池，每个训练问题4个token，共12044个，锚定原GMM中心。保留簇样本数和有效拟合rank；错配基用固定derangement，只换基不换中心。拟合的是重构方向，不是重新拟合GMM。

主功能比较：prefix16、最后4位置，delta_local8相对shared8与zero_update的KL、参考NLL。KL衡量保真度，不等于纠错；保留完整输入不能称为用8维压缩整个状态。全词表log-probabilities与实际bf16替换前后向量保存，独立重算。

## 执行与交付

配置 configs/delta_online_pilot_20260925.json；根目录 /lambda/nfs/dami/hss/delta-online-pilot-20260925。本地测试→GitHub→Lambda fetch/ff-only。单GPU；CPU BLAS最多4线程；不升级环境、不改旧实验。

自动队列：准备与冻结 → 真模型功能冒烟 → 冒烟审计 → 补齐前缀confidence → 风险拟合 → 风险独立审计 → 固定功能批次 → raw审计 →统计与报告。源文件冻结后不原地修改，失败保留日志，另开恢复命名空间。报告生成之后还要实际查看PNG/浏览器和检查链接，delivery.json才代表交付完成。

预估从实施到首轮交付3–5小时；不根据部分测试正负结果改变范围或追加搜索。自由生成Δh steering尚无冻结策略，本轮不自动启动。
