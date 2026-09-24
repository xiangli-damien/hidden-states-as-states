# GSM8K test 两模型全量采集完成

完成时间：2026-09-23 21:54 UTC。数据版本为 `openai/gsm8k`、`main`、`test`，revision `740312add88f781978c0658806c59bc2815b9866`，每模型完整1,319题。

| 模型 | 题数 | 本次生成判正确 | 正确率 | 生成token | 存储（GiB） | 结束原因 |
|---|---:|---:|---:|---:|---:|---|
| Qwen2-7B-Instruct | 1,319 | 1,134 | 85.97% | 341,885 | 53.14 | EOS 1,317，长度上限2 |
| Meta-Llama-3-8B-Instruct | 1,319 | 1,083 | 82.11% | 259,705 | 51.83 | EOS 1,319 |

正确率指本项目零样本CoT、固定prompt/chat template、greedy生成及GSM8K答案解析器下的结果，不是官方不同prompt协议的benchmark分数。生成上限2,048 token。Qwen已有合格全量数据经核验后复用，未重复采集；Llama本轮补齐。长度截断样本原样保留，不补采或删除。

## 保存内容与核验

- 每题原prompt、完整生成、token IDs、gold及独立复核的正确性标签。
- 教师强制前向得到的全层全生成token、prompt-last和mean，另存最终RMSNorm之前/之后的对应表示。
- Qwen hidden层轴29×3584，Llama33×4096；Zarr存储float32，模型推理bf16。
- 两模型各42个分片。文件SHA全部重读核对，题ID顺序、无缺失/重复、数据/模型revision、prompt/chat模板、提取方式及tensor shape/dtype均通过。
- 独立重新解析每题答案，与已有标签及extracted answer一致。
- 因果重放沿用并验证采集时各分片的检查收据，本轮最终CPU审计未重新运行GPU因果重放。
- Llama最终全文件/标签审计386.97秒。

## 数据入口

统一完成清单：`/lambda/nfs/dami/openact/runs/gsm8k_test_20260923_v2/collection_manifest.json`。

- Qwen：`/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923/qwen2`
- Llama：`/lambda/nfs/dami/openact/runs/gsm8k_test_20260923_v2/llama3_full/llama3`
- Llama审计：`/lambda/nfs/dami/openact/runs/gsm8k_test_20260923_v2/llama3_full/full_test_audit.json`
- Qwen审计：`/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923/full_test_audit.json`

清单SHA：`28c06134981c81ec8c70021af61f0f41df92416d00b81380220d2a86c8cd820d`。
Qwen审计SHA：`68764e8780d8892104057ac7a12f4118b130f0b14911cf9697a068950495be61`。
Llama审计SHA：`af3d6e53a114827fe0017499becec7693687f236c31dd64b1ede2f817758915e`。

本地仅同步轻量清单和审计到HSS `results/gsm8k-test-collection-20260923`，约105GiB激活仍保存在dami。代码来自OpenAct隔离分支 `codex/gsm8k-test-qwen2-llama3`，实际采集源码 `bff452b`；初始失败尝试保留。收集任务释放GPU后，后续公平FA/MFA及能力门槛后的记号交换按队列顺序执行。
