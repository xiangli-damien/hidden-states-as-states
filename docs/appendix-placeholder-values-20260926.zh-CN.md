# Appendix placeholder values — verified 2026-09-26

这里只列填空内容与必须区分的版本。当前 OpenAct/HSS 的参数不能自动认证旧 Table 1/Table 2 的全部数字。原附件尚未改写。

## A, line 42 — answer extraction and matching

以下对应当前 OpenAct 实现：

```latex
for MATH, the last brace-balanced \texttt{\textbackslash boxed\{\}} expression, with explicit-answer/conclusion and last-number fallbacks when absent, followed by normalization and exact, numerical, or symbolic-equivalence matching; for MMLU and Belebele, an extracted option letter A--D matched exactly to the reference; for TheoremQA, the first boxed expression, otherwise an explicit answer/conclusion or type-dependent fallback, normalized and compared according to the annotated answer type (Boolean, option, number, or list). Numerical matching uses absolute tolerance $10^{-6}$ and zero relative tolerance
```

特别注意：MATH 没有 boxed 不等于自动判错；TheoremQA 当前是 first boxed，不是 last boxed。不能把格式完整率等同于正确率。

## A, line 45 — prompt templates

括号内指向下面新增的段落：

```latex
see the task-specific prompts in Appendix~\ref{app:task_prompts}
```

对应的可直接插入 LaTeX 段落已另存为 `app_prompt_templates_verified.tex`。

下面是实际模板文本，变量由数据集填入，外层由模型原生 chat template 包装。MMLU 的 question 字段包含问题和 A–D 选项；Belebele 的 question 字段包含原语言篇章、问题和选项。GSM8K64 功能验证采用下面的 MATH 模板，而非 OpenAct 常规 GSM8K 模板。

### MATH / GSM8K64 functional tests

```text
Question: {problem}
Please reason step by step, and put your final answer within \boxed{}.
```

### MMLU

```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Question:
{question}
```

### Belebele

```text
Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering.

{question}
```

### TheoremQA

```text
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float], The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float], the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).
You need to output the answer in your final sentence like 'Therefore, the answer is ...'.

### Question:
{question}

### Answer_type: {answer_type}

### Response:
```

旧 JBB 采集代码把原始 attack/behavior 文本作为 user message，没有追加以上 CoT 指令。因此附件的 “All tasks use a zero-shot chain-of-thought prompt” 应限定到 reasoning/capability tasks。HarmBench 的原始运行模板尚未核实，不能代填为 JBB 模板。

## A, line 45 — maximum new tokens

当前 OpenAct MATH/MMLU/Belebele 收集及 steering 可填：

```latex
2{,}048
```

这不是已证实的所有历史任务统一上限。旧 JBB collector 的 safe/risk 配置默认上限分别为 512/1024，原历史运行目录未找到；不能把默认值当作当时的实际设置，也不能声称旧 safety 一律为 2048。

## A, line 74 — GMM parameters

当前 Qwen MATH response-mean 地图可填：

```latex
$k$-means++ initialization, three initializations per candidate, a convergence tolerance of $10^{-5}$ in mean log-likelihood, additive diagonal covariance regularization $10^{-6}$, at most 2{,}000 EM iterations, and base seed 42
```

历史 response-mean 拟合的派生种子为 $42+1009\ell+10007r$，$r=0,1,2$。本次新 reliability/pooled control 使用保存协议中的独立派生种子和单次 D-squared k-means++ seeding，不能称为 sklearn 完整 KMeans 拟合。

**不能整篇统一使用这一组参数。** 已保存的默认 Llama-3.2 MATH 地图为 2 次初始化、最多 200 次 EM，tol=1e-5、reg=1e-6、base seed42。旧 Table 1 的 Qwen 代码为 sklearn k-means 初始化、3 次、最多300次、tol=1e-3、reg=1e-4，种子42+1000*layer+K，且先标准化。旧 JBB 代码则是3次、最多200次、reg=1e-2。

## A, line 84 — default ICL tolerance

已核实的当前默认 Llama 地图、Qwen characterization/steering 地图为：

```latex
2\%
```

新 reliability 同时保留 0% 和 2%，primary=2%。但这不能无条件补成 “all other maps use 2%”：旧 JBB 默认是4%；旧 Llama3 MATH prediction 汇总为2%；旧 Qwen notebook 中不同 Table 1 数字对应1%/2%行，尚不能唯一对应表中整对数字；原 Qwen monitoring 拟合文件未保存。因此应逐实验限定，不能制造一个统一默认值。

## A, line 118 — continuous-baseline parameters

下面对应能够追溯到旧 Table 1 数字的 LMD baseline 代码：

```latex
training-set feature-wise standardization using $(x-\mu)/(\sigma+10^{-8})$; LDA with the SVD solver and no shrinkage; Gaussian NB with \texttt{var\_smoothing}$=10^{-9}$; logistic regression with $C=1$, L-BFGS, balanced class weights and at most 1{,}000 iterations; linear SVM with $C=1$, balanced class weights, at most 2{,}000 iterations and three-fold sigmoid calibration; and a one-hidden-layer MLP with 128 ReLU units, AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$), batch size 128, at most 20 epochs, and early stopping after five non-improving epochs on an internal 10\% validation split (seed 42)
```

必须同步把此前一句 “concatenated prompt-last vectors of all layers” 改成 “last-layer prompt-last vectors”，才能对应这些历史 baseline。所有层拼接是讨论过的新协议，尚未运行，不能只改描述而沿用旧表数字。当前 HSS 的 MLP 是另一套128/64双隐藏层配置，不能用来补旧 Table1。

## A, line 123 — sentence segmentation

当前 HSS 可填：

```latex
periods, question marks, or exclamation marks followed by whitespace or end-of-text, Chinese sentence-final punctuation, and runs of newline characters; character boundaries are mapped to the end of the first whole generated token covering the boundary, duplicate boundaries are removed, and the final boundary includes EOS when present
```

准确正则为 `[.!?](?=\s|$)|[。！？]|\n+`。这是当前复现代码的明确规则；原 Qwen Table2 模型和配置未保存，不能据此认证其历史分句和所有指标。

## B, line 16 — pooled-layer GMM

已完成并通过独立审计：145,000 × 3,584 原始向量，29层，每层5000题；K=2…80，3次初始化，diag float64；79个候选、237次初始化全部收敛。

| ICL tolerance | Selected K | Sample-weighted layer purity, nearest center | Sample-weighted layer purity, posterior MAP |
|---:|---:|---:|---:|
| 0% | 80 | 0.5660207 | 0.5517862 |
| 2% | 70 | 0.5224828 | 0.5156000 |

按2% ICL、论文最近中心分配，两个空依次填：**`70`**, **`0.5225`**。这是按题目-层向量数加权：sum_c max_l n(c,l) / 145000，不是各簇 purity 简单平均。

原句“主要区分层而不是样本”的推断过强：52.25%说明部分按层集中，同时仍有大量跨层成员。建议直接替换整个结果句：

```latex
Within the searched range $K\in\{2,\ldots,80\}$, the 2\% ICL rule selects 70 clusters with sample-weighted mean layer purity 0.5225 under nearest-center assignment. The pooled clusters show partial separation by depth while retaining substantial cross-layer membership. We use per-layer fitting to characterize sample partitions conditional on layer. The strict ICL minimum occurs at the upper search boundary ($K=80$), so the selected count is conditional on this search range.
```

如果使用与325题 sink相同的posterior-MAP口径，纯度应填 **0.5156**，并相应改写assignment说明。不能混用两个值。

结果根目录 `/lambda/nfs/dami/hss/appendix-global-gmm-20260926`；独立审计核对全部79个候选收据、模型哈希与ICL算式、完整纯度列联表，并对每个选中模型跨29层的290个向量独立重算分配。轻量结果已同步到 HSS `results/appendix-placeholders-20260926/global/`。

## B, line 21 — eta sensitivity (computed)

固定已有 Qwen MATH5000 的29层、2% ICL地图，只重跑相邻层余弦 Hungarian 对齐：

| eta | Global states | Mean self-transition, nearest center | Mean self-transition, posterior MAP |
|---:|---:|---:|---:|
| 0.3 | 84 | 0.5701143 | 0.5669286 |
| 0.4 | 84 | 0.5701143 | 0.5669286 |
| 0.5 | 84 | 0.5701143 | 0.5669286 |
| 0.6 | 84 | 0.5701143 | 0.5669286 |
| 0.7 | 84 | 0.5701143 | 0.5669286 |

按论文最近中心的分配定义，四个空依次：`84`, `84`, `0.5701`, `0.5701`。更自然的整句：

```latex
Varying $\eta\in\{0.3,0.4,0.5,0.6,0.7\}$ leaves both the number of global states (84) and the mean adjacent-layer self-transition probability (0.5701) unchanged.
```

这里包含 embedding→第一层及末block→final-RMSNorm后两个边界，总共28条相邻层边；不是某个 sink 的单簇自转移率。若沿用干预历史报告的 posterior-MAP assignment，最后两个空应为 `0.5669`，不能混写。

独立 float64 检查也发现：637条候选匹配中，没有余弦落在[0.3,0.7)；保留匹配的最小余弦为0.91098，拒绝匹配的最大余弦为0.22185。因此这些阈值确实不会改变匹配结果。

## C, line 80 — actual examples from cluster 8 (K33 at layer14)

下列两题都核实属于该325题簇；都是原始历史回答，不是 steering 后生成。省略号代表前文省略；正文尾部保持原意和数值。

```latex
\par\noindent
\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{
\textbf{No final answer.} Intermediate Algebra, Level 2
(\texttt{math\_2741}).\\
\ldots{} Given the constraints of this response format and the educational
goal, the focus should be on understanding the method rather than the
exact numerical solution, which would require computational tools or
further algebraic steps not detailed here.
}}
\par\medskip\noindent
\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{
\textbf{An unboxed answer.} Algebra, Level 5
(\texttt{math\_723}).\\
\ldots{} \texttt{\#\#\# Final Values:}\\
Thus, $A=-2$, $B=-\frac{1}{2}$, and $C=1$.\\
\texttt{\#\#\# Summing} $A$, $B$, \texttt{and} $C$:\\
$A+B+C=-2-\frac{1}{2}+1=-\frac{3}{2}$\\
Therefore, the sum $A+B+C=-\frac{3}{2}$.
}}
```

第一题参考答案3/2，回答没有给出所求第三根；第二题明确给出-3/2但未boxed，参考答案是-3，因此这条并非“只缺boxed的正确答案”。它展示的是有答案但未boxed的另一种组成。

另外，325题/9正确是 posterior-MAP 分配的计数；同一组中心改用最近中心分配后，该簇为345题/5正确。这两条示例在两种分配下均属于cluster8。正文若统一采用最近中心，不能继续把325/9声称为最近中心的计数；干预的历史成员选择确实采用MAP，应单独说明。

## Evidence

- OpenAct task templates: `packages/openact-core/src/openact_core/tasks/prompts/`.
- Parsers/matchers: `packages/openact-core/src/openact_core/tasks/parsers/`; `packages/openact-eval/src/openact_eval/matchers/`.
- HSS map configs: `configs/qwen_math_raw.toml`, `configs/base.toml`, `configs/qwen_math_reliability_20260925.json`.
- Historical Table1 evidence: HSS `results/chapter5-code-audit-20260926/legacy_aurpc.py`, `legacy_jbb.py` and `remote_source_receipts.json`.
- Alignment, exact full source responses, and source SHA256 receipts: HSS `results/appendix-placeholders-20260926/alignment_and_examples.json`.
