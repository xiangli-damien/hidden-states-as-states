# 冻结 prompt-last 地图：几何重构和模型内替换

用户在2026-09-24授权尝试。新根目录 `promptlast-replacement-20260924`，既有mean实验保持不变。

## 定义与范围

重构是把对象x编码为z，再解码为x_hat。逐token向量、句子均值、prompt-last均可成为对象。只重构句子均值，不能宣称重构整段逐token序列：许多不同序列有同一个均值。把重构向量放回模型后比较输出，是重构的功能保真检验；不是定义本身，也不是正确率改善。

## 查看新结果之前固定的协议

- Qwen2-7B-Instruct固定revision，原block14输出，raw3584维，bf16/SDPA。
- 复用 `revision-foundations-20260923/geometry/p0_l14_last` 原GMM及原局部方向，K32，3011道MATH训练题，原ICL网格1/4/8/16/32/64。K不是2%容差的全回答mean地图K33。
- 按原最近中心分配，中心、方差、权重和local basis均冻结。local8用原32方向中前8个；新拟合的shared8仅用同3011训练题相对所属GMM中心的残差，避免局部/共享的中心不同。三个wrong-local置换seed42/137/271，保持recipient中心。
- 8条件：identity、所有题共用训练均值、GMM中心、中心+shared8、中心+local8、三种wrong8。global-mean是无区域ID对照；local/shared都有同一ID及8连续坐标，但字典总存储不等。
- 使用上一轮相同96题：历史MATH测试32＋功能GSM64。MATH不在地图/方向训练集；GSM不参与拟合，但两组都已被研究使用，属于探索性比较。
- 每题只替换完整chat prompt的最后一个token，位置为len(prompt_ids)-1。没有未来token参与编码；每条件新cache、只patch一次。其他prompt位置保持原样，不称整道题压缩。
- 几何：重构平方误差、相对训练均值的NMSE。功能：KL(p_original||p_patch)、首token/前16token/第17token起/完整原回答NLL。参考是原模型实际回答，包含EOS，不是标准解答。
- 主比较GSM local8−shared8，KL和全参考NLL一起报告。另报local8−centroid、local8−wrong平均、centroid−globalmean。2000次问题配对bootstrap，pointwise95%，无训练不确定性或多比较校正。不得跨本次width1和旧width16直接称地图优劣。
- 描述区域占用、最近中心距离及源验证集q95覆盖；不按目标表现筛题或选参。

先独立算子测试和真实MATH/GSM各一题冒烟，独立审计通过后执行固定96题。完整逐题向量、理想及bf16实际patch、logprob、参考token损失、原文、统计和图均保存。源码先GitHub再Lambda fetch；单GPU，CPU4线程。预期只需分钟级模型前向；含准备、核验及交付暂估20–30分钟。
