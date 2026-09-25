# Response-mean 退化输出区域的方向干预：有限初测

## 范围与先验检查

用户于 2026-09-25 授权检查并补充“沿 HSS 方向离开 sink”的小实验。仅此新实验；旧队列不重启，不改论文原文件，不扩大层、rank 或模型搜索。

当前 ICLR 源稿称 sink 回答“almost entirely empty or malformed”；旧 ICML 源稿举 global ID3，但没有找到能将这个 ID 对应到本次保存地图的证据。**因此本实验不能称为原文 sink 的精确复现。** 当前 Qwen2 MATH5000 没有空生成，82条在2048token截断。当前 block14、2%ICL容差、K33 的完整回答均值 GMM 中，local cluster8 有325题，原自动正确9/325，训练/开发/测试成员197/73/55。

抽查固定种子20260925的8条原文：math_2741、2570、2675、2482 等确实出现长篇解释后不提供具体解答；math_723 有未boxed的数值结论；math_2776、599 有完整答案。这是混合组成，不能写成全部空白、全部格式失败或全部错误。重复率和“超出范围”等词只作描述性启发式，不当作人工语义标签。

## 冻结方案

- Qwen/Qwen2-7B-Instruct，revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`；HF、BF16、greedy、SDPA、batch1，最多2048新token。
- 层定义：`hidden_states[14]` = 第14个 transformer block 的输出 = `model.model.layers[13]`。不是零起算的 `layers[14]`。
- 保留原完整回答均值 GMM，不重拟合、不归一化。原图是全部5000题无标签拟合，**transductive**；不能称未见过这些问题的新测试地图。
- 用原train3011独立确定候选：成员至少50、完整非空boxed率<=30%，取最低boxed率，平局取最多成员。不用正确性选择方向。
- `d = mean(train非候选簇的回答均值) - mean(train候选簇的回答均值)`。不用包含测试数据的GMM中心当方向锚点；两个均值都只算train。
- dev16题从原validation中按hash抽取。alpha={0.1,0.3,1.0}，仅根据dev完整boxed比例选择；平局选较小alpha。不因没有收益扩大搜索。
- 原test候选簇全部55题（上限64）；另按hash取100道非候选、原回答有完整boxed且未截断的问题检查副作用，不要求其原答案正确。
- test每题三组：zero、alpha*d、同alpha且等范数的固定随机方向（种子20260925）。随机方向不经筛选。
- dev64次 + test465次，共 **529次完整生成**，另有两题32token的真实模型冒烟。计划6小时GPU上限，若未完成保留部分结果并说明，不在部分结果上宣布成功。

## 干预位置与它实际检验的内容

每一组都从同一题的原prompt重新生成，KV缓存全新。prompt prefill不加向量。从第一个生成token被送回模型计算起，对其block14输出加同一向量，再让后续层和解码继续。每个被前向处理的生成token都修改；首个输出token仍由未改动的prompt产生。最后刚采样出来的token/EOS不再前向，因而不在hook统计中。

对一组**固定的**向量加d，均值会精确平移d（实际BF16另有舍入）。但自由生成时，上游状态、token内容与长度都会改变，**最终整段回答均值不会保证只平移alpha*d**。本实验测方向的行为效果，不能用这条代数等式直接宣布“已离开sink”。保存每步扰动范数、舍入误差、方向投影和实际前后状态均值，便于检查。

## 评分与成功标准

现有OpenAct MathParser存在“取全文最后一个数字”的fallback，不适合作为“是否真正给出最终答案”的主终点。

因此本次主终点明确缩窄为：**输出存在括号完整、内容非空的 `\\boxed{...}`**。这是可重复计算的格式遵从指标；它既不是所有未boxed答案的完整解析率，也不是正确率或推理有效性。正确率使用冻结的OpenAct自动评分作为次终点；另保存宽松parser成功率、长度、截断与完整原文，所有变化题需后续检查其是否真的提供答案。

主比较是55题候选test的HSS−random完整boxed比例。成功需要：题目级配对bootstrap95%区间下界>0、单侧精确McNemar p<.05，且HSS比zero的点估计提高。正确率与正常题副作用另外完整报告，不从其他终点挑“成功”。只有一个固定随机方向，结果不能推广为优于所有随机方向。

该队列仅在dev选择alpha；test生成结果不反馈调参。地图和历史标签早已被分析过，候选检查也发生于看过总体描述之后，所以即使新干预test分离，也只能称**探索性历史数据初测**。

## 文件、运行与审计

- 配置：`configs/sink_direction_20260925.json`
- CPU准备：`.venv/bin/python scripts/prepare_sink_direction.py --config configs/sink_direction_20260925.json`
- GPU冒烟：OpenAct `.venv/bin/python scripts/run_sink_direction.py --root /lambda/nfs/dami/hss/sink-direction-20260925 --smoke-only`
- GPU队列：同一命令去掉 `--smoke-only`。
- 原始结果：`/lambda/nfs/dami/hss/sink-direction-20260925`；每题每组JSON、扰动统计NPZ、SHA收据；方向、组成表、输入、plan、selection均保存。
- `status.json` 是真实进度；`COMPLETE.json`/`summary.json` 只有全量成功后生成，不能把代码准备完成当实验完成。
- 本地不含torch，CPU测试通过后，GPU环境需补跑hook测试；真实模型检查zero与无hook逐token完全相同、首token不变、每个后续位置都被修改。
- 每30分钟检查现有heartbeat；健康无变化时不重复提醒。完成后独立复算计数、配对统计、检查改变原文，再交付结果。

### 实际启动记录

源码 `7329ba2` 已本地测试后推GitHub，Lambda已fetch/ff-only。GPU环境4项测试全部通过。真实Qwen两题各32token冒烟完成，zero与无hook生成完全相同；两个非零方向均覆盖31个已前向的生成token，首token不变。alpha0.3的理想扰动范数1.9707，实际HSS约1.973–1.976、random约1.973；显存峰值14.214GiB。

正式进程 **480617** 已启动。冻结plan SHA `ef04d559b77a1c014de33dfc061c59230e470b3057c684c1126a66da5cef857f`。前8次完整生成共6592token，耗时164秒，约40token/s；这仅用于时间估计，不据此报告效果。独立复算脚本 `scripts/audit_sink_direction.py` 将在完整结果后运行，随后还需逐题检查变化内容。请以远端status为最新进度。

代码仍按本地→GitHub→Lambda fetch/ff-only部署，作者Xiang Li；不传源码覆盖远端、不改旧冻结文件。

## 时间与截稿

初步估计3–6小时，需以冒烟和dev实际吞吐更新。ICLR官方正文截止2026-09-25 23:59 AoE，即纽约2026-09-26 07:59；官方讨论期修改截止为11月18日，不是11月27日：

- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines

论文按尚无该实验正结果来写；即使格式成功，也不自动证明离散、通用、可控的推理状态。
