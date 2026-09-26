# 第五章代码与历史结果核对：2026-09-26

## 结论与证据范围

**当前实现的12个问题可以查清；原稿 Table 1、Table 2 的全部历史数字不能据此自动认证。** 本次核对当前HSS、Lambda保存的Llama-3.2运行、旧LMD代码与notebook输出，以及用户上传ICLR稿件的第7、8、19页。没有改评分、阈值或训练配置，也没有启动新实验。

最重要的发现：

1. 当前五种连续基线默认全都只用最后一个选定层的 `[N,d]`；HSS用全部选定层状态。旧Qwen notebook也明确打印第28层、3584维，多个基线数字与Table 1吻合。不能把这写成已控制输入层数的纯离散化比较。
2. 当前HSS-NB为每层完整地图词表、alpha=1、经验先验、log-space求和、归一化posterior。但仓库旧NB和LMD旧NB的词表／先验／未知状态细节不同。
3. 当前monitoring默认40/20/40，以完整回答分组；FAR包括final，early detection排除final。稿件正文与附录恰好对FAR写法不一致。
4. Saved是**每条失败回答的潜在节省比例的平均**，没有检测到的为0，不是仅在detected上平均，也不是按所有token数加权。
5. `0.500 → 0.702 → 0.771`尚未获得同一实验的逐层结果支持。旧逐层NB代码用累积层序列，不是单层NB。固定prompt末token的embedding常量也可能直接产生0.5，不能据此排除输入语义／难度解释。

## 1. Table 1 continuous baseline input（P0）

当前入口：[runner.py](../src/hss/experiments/runner.py)，`_continuous`第99–105行；所有LDA、GaussianNB、Logistic、LinearSVM、MLP都走同一分支（第157–178行）。默认 `continuous_features="last_layer"`，来自[配置](../src/hss/experiments/config.py)第81行和[base.toml](../configs/base.toml)第46行。

- 默认形状 `[N,d]`，取所选层列表最后一层；选全层时就是模型最后层。不是自动选择best layer。
- 显式设为`all_layers`才得到`[N,(L+1)d]`。Qwen全29个存储层为103936维；默认3584维。
- 五个方法使用相同输入。默认训练集拟合StandardScaler，并同样变换验证／测试集；这与GMM使用raw、无标准化是两个不同设置。
- 旧 `experiments/prediction.py` 第67–73、115–129行也如此。

**实际旧Qwen证据：** Lambda `cluster/aurpc.py`第1840–1842行固定`probe_layer=N_layers-1`；对应`aurpc.ipynb`第42个零基cell输出“Using layer 28 hidden states (dim=3584)”。训练2000、测试2934。Logistic、SVM、LDA、GaussianNB的AUROC/accuracy分别为`.7179/.6629`、`.7219/.6602`、`.6294/.6057`、`.7372/.7048`，与稿件四舍五入后的数值一致。旧JBB代码第1761–1764行也固定末层。

可以写“last-layer continuous probes versus an all-layer discrete sequence predictor”。若要写“isolating the effect of discretization”，还缺同输入层范围的连续基线对照。现有数字不能通过改caption变成全层concat实验。

## 2. HSS-NB smoothing与unseen state

当前 `CountNB`：[evaluate.py](../src/hss/experiments/evaluate.py)第43–73行。每层词表来自冻结地图所有成分对应的global ID（runner第481–488行），不只来自该类已出现的状态。

对该层地图中的状态：

\[
\widehat p(s_\ell=k\mid y)=\frac{N_{\ell,k,y}+1}{N_y+|V_\ell|},\qquad
\widehat p(y)=\frac{N_y}{N}.
\]

类内未见状态的计数为0，仍获得正概率。**地图词表外的新ID会触发KeyError**，没有UNK回退；正常冻结地图assign只会产生词表内ID，因而不会需要这个回退。不要写“任意未知ID都自动加一处理”。

旧实现有区别：

- `src/hss/predict.py::NaiveBayesClassifier`：使用训练样本中最大global ID+1作为所有层共享词表大小，连类先验也加alpha；范围外ID忽略该层证据。
- LMD `aurpc.py::NaiveBayesTrajectoryClassifier`：同样使用全局词表，先验为经验频率；超出词表范围时，两类都加`log(1/V)`，不改变该层log odds。

因此，不能把当前逐层`|V_l|`公式不加说明地声称为所有历史数字使用的公式。

## 3. HSS AUROC scalar score

当前runner第137–145行使用 `predict_proba(states)[:,1]`。类1默认是正确／安全，NB内部先累加类log分数，再通过logsumexp归一化；没有直接连乘概率。Monitoring第199行用 `1-p(class1)`，即失败风险。

旧 `experiments/prediction.py`第48–51行和旧LMD notebook的NB AUROC同样使用类1posterior。不是仅使用`log p(s|y=1)`。使用类log分数之差在数学上与posterior排序等价；有限精度下posterior可能饱和成0/1并产生并列，所以若改为log odds，应保存为明确的新评分版本并重新计算相关指标。

## 4. Table 1 accuracy阈值

当前runner第180–185行：

- HSS-NB、LDA、GaussianNB、Logistic、MLP：posterior **>=0.5**。
- LinearSVM：`decision_function` **>=0**。当前未进行SVM概率校准。
- 旧HSS接口调用`argmax`；精确平票时选类0，与当前`>=0.5`选类1有细微区别。

**更早LMD代码不能套用当前SVM定义：** `aurpc.py`第1516–1539行和`jbb.py`第1027–1044行使用`CalibratedClassifierCV(cv=3, method="sigmoid")`；SVM的概率转log odds后交给`compute_metrics`。后者第1742–1749行按分数是否落在[0,1]内推测它是概率还是logit。这是一个脆弱的旧规则：若一组logit碰巧全部在[0,1]，会被当成概率。当前runner已使用显式方法阈值，但不能据此替旧accuracy认证。

正文caption应区分概率与decision score；对于历史表格，必须按实际产生该数字的那次运行说明是否校准。

## 5. §5.2 train / validation / test（P0）

当前默认40%/20%/40%；训练／holdout使用split_seed，验证／测试再使用split_seed+1，均以完整回答group_id分层划分。预测任务则40%/60%，没有验证部分。

**操作顺序需要写准确：** [openact.py](../src/hss/data/openact.py)先无标签地准备所有回答的prefix缓存，再由`grouped_split`把同一回答的全部prefix放入同一split。不是字面上先split再产生prefix，但组间隔离仍成立。地图、预处理和NB只用训练部分拟合。

实际保存的Llama-3.2监控运行`70844a0e5f504353b4d793fd`已经核对：2000/1000/2000条回答，交集为0；训练prefix数48271。因此连续probe训练矩阵为`[48271,2048]`。训练中每个prefix是一条记录，长回答的边界更多、贡献更多计数；NB先验也是训练prefix行上的类频率。

**原稿Qwen Table 2的独立运行文件尚未找到；40/20/40不能仅凭当前默认值认证为原稿实际划分。** 用户此前也确认原Qwen前缀地图、NB及阈值未保存。

## 6. False alarm是否包括final（P0）

当前 `far_scope="all_boundaries"`，校准和测试FAR都包括final。`far_scope="nonfinal"`是已支持的另一种口径；它在校准与评估两处都排除final。没有非final边界的成功回答仍保留在FAR分母中。

阈值只从成功验证回答的最大eligible边界风险分数校准，报警规则为 **score > threshold**。校准目标为验证FAR不超过10%，不是保证测试FAR不超过10%。

稿件第8页正文写any boundary，第19页D.2却写any non-final boundary，并声称两者一致：这是实际矛盾。若最终采用`b < B_n`，需对每种方法重新校准验证阈值并重算测试结果；不能只改正文而保留12.8、9.9等数值。

## 7. Detected

[evaluate.py](../src/hss/experiments/evaluate.py)第124–155行：一个失败回答只要有至少一次**非final**报警即detected；同一回答多次报警只计一次。用于Saved的是第一个非final报警边界。只在最终边界报警不算early detection。

## 8. Saved的分母（P0）

当前第138–159行实现：

\[
\mathrm{Saved}_{failed}=\frac{1}{N_{failed}}
\sum_{n:y_n=0}\mathbf1\{\text{early detected}\}
\left(1-\frac{t_{first,n}}{T_n}\right).
\]

未提前检测的失败回答贡献0。这是逐回答比例的平均，不是只对detected failures平均，也不是把所有失败回答的节省token数相加后除以总token数。实现另存`saved_token_fraction_all`，不要与`..._failed`混用。这里是若立即停止的**潜在token节省**，没有减去监控计算成本，也没有执行纠错重试。

当前定义已通过运行行为核对；原稿32.2是否确实出自同一分母仍缺Qwen逐回答记录，不能只凭数字相近认证。

## 9. @50% AUROC

按token进度，不按句子序号。取`token_end <= 0.5*n_tokens`的最后一个已有边界；不是最近边界，也不是第一个越过50%的边界。半程之前没有边界的回答从该AUROC中排除，同时报告`half_coverage`。这是利用最终长度定义的离线评估时点，不是模型提前知道最终回答长度。

## 10. Prefix entropy

[openact.py](../src/hss/data/openact.py)第211–229、247–249行：若提供对齐的逐token熵，先计算累计和除以已生成token数，再取当前边界处的值，即**prefix token entropy mean**，不是边界的下一token熵。

具体“每个token熵”对齐哪个预测位置，还必须读取上游保存该数组的配置；该reader只规定平均方式。默认`token_entropy_path=null`，实际Llama监控将其报告为unavailable，没有拿全回答平均值伪装为prefix值。原稿Table 2该项还缺原始token数组及提取口径。

## 11. Prefix logprob

同样为前缀已生成token的**平均log probability**，不是累计值、不是仅边界token的logprob。runner第223–235行取负号作为风险分数（越负的平均logprob对应越高风险）。默认数据路径为空，当前已有Llama监控也报告unavailable；不能据此认证原稿那一行。

## 12. Continuous prefix probe（P0）

当前monitoring强制方法列表为HSS-NB和Logistic。Logistic默认只使用**最后层的prefix-average向量**，训练集StandardScaler，`C=1`、`max_iter=500`；不是all-layer concat，不是SVM。HSS却使用全部选定层prefix states。

因此“same prefix information”可以表示两者都只看已生成前缀，但不能据此写成“identical layer inputs / isolating discretization”。原稿Qwen具体probe配置仍须原始运行来源确认；已核对的Llama运行不能充当Qwen Table 2。

## 13. Table 1与逐层数字的历史来源差异

### 已找到的Qwen notebook

`aurpc.ipynb`保存Qwen2-MATH、prompt-last、2000训练/2934测试的输出；不是当前完整5000条上的2000/3000。

| 来源 | HSS AUROC | HSS accuracy | 备注 |
| --- | ---: | ---: | --- |
| notebook主要结果 | 0.7614 | 0.7021 | 冻结选择显示2% |
| notebook容差扫描1%行 | 0.771176 | 0.709952 | 与0.771相关，但accuracy不是0.702 |
| notebook容差扫描2%行 | 0.762637 | 0.702113 | 与主要输出也略有不同，保留差异 |
| 稿件Qwen-MATH | 0.771 | 0.702 | 未找到同一结果行完整匹配 |

这些发现不证明论文数值必然错误，但明确说明不能声称已找到完整对应运行。需确认数字来自哪个选择版本／记录，不能跨行拼接。

### 同名目录是另一个模型

`cluster/cluster_trajectory_icl_results/final_summary.json`目前记录Llama-3-8B-Instruct、33存储层、4096维、2000/2934、2%。其HSS AUROC/accuracy为`.750747888/.708248125`，与稿件Llama3-MATH的75.1/70.8相符。

同目录`layer_wise_comparison.csv`属于这份Llama结果：layer0/1/32的NB AUROC为`.500000/.695038/.750748`，不能拿来验证Qwen的`.500/.702/.771`。同名目录没有模型命名空间，当前磁盘表格与notebook保留输出不能混用。

旧逐层函数明确用`X_traj[:, :layer+1]`，所以曲线是**累积层NB**；连续probe才是每次单独取当前层。没有查到可认证所提三点的同一份Qwen逐题预测文件。

此外，即便三点将来得到核实，它也只说明经contextualization后的表示对最终结果有预测信息。prompt最后一个token常是固定模板token；若embedding相同，0.5来自常量输入。第一层attention开始读取题目后，语义、题型、难度都可能进入表示，不能据此直接排除input similarity。

## 14. 可以现在改的文字，与需要重算的内容

- Label-free仅指地图拟合、选K及匹配不使用outcome标签。标签可用于预先分层划分；冻结地图后再估计NB的监督计数。不要将整个预测方法称为label-free，也不要将“无梯度优化”写成“没有从标签估计的参数”。
- Safety可写competitive，而非优于全部trained probes；Table 1需明确last-layer continuous comparator。
- Table 2可以如实陈述该operating point检测率更高、同时测试FAR更高，不能从中推出阈值更稳定。
- 改FAR为nonfinal、改连续probe为all-layer、改旧NB词表／先验或打分，均是新口径，需保存新版本并重算；本次没有执行这些变更。
- `.500→.702→.771`先不写成已核实逐层结果；先确定它是单层还是累积层、模型、split和ICL版本。

## 审核文件

本地`results/chapter5-code-audit-20260926`保存：当前源码哈希及CPU行为核对、Lambda两次实际运行的配置／split计数／指标收据、旧LMD源码快照与notebook文本输出、旧逐层CSV／容差扫描／final_summary，以及用户PDF哈希。完整文件不含模型权重与激活。

行为检查实际验证了末层与concat输入维度、词表内未见状态／词表外ID、首次报警、Saved分母、两种FAR口径及半程边界；两项既有相关测试通过。这些是实现核对，**不是新的论文实验结果**。
