# HSS / OpenAct：完整存档交付与恢复说明

**2026-09-26 已完成。** Cloudflare R2 总完成标记于 **15:01:22 UTC** 发布；15:31 UTC 独立复核通过，上传进程已退出。原始文件全部保留，Lambda 实例未关闭。

## 1. 存档内容与位置

| 存档 | 已核验范围 | R2 位置 |
|---|---|---|
| 本次完整研究存档 | 185 包；1,164,373 个文件/条目；原文件 774,348,166,982 bytes；tar 对象共 776,007,567,360 bytes | `s3://autoact-data/hss/research-archive-20260926/` |
| 已有 MATH / MMLU 原始数据 | 57,126 条回答、573 分片，约 2.21 TB 原文件 | `s3://autoact-data/openact/math-mmlu-full-20260916/` |

本次包括所有 HSS 实验、拟合参数与结果，OpenAct 新增/辅助采集（包括 BELEBELE、GSM8K），拟合输入缓存，Git 源码快照及历史 bundle，以及本地报告和用户提供的附录。旧 MATH/MMLU 数据在总索引中引用，不重复上传。模型下载缓存、虚拟环境和凭据不包含在存档内。

本地独有报告/附录快照含 59,852 个文件、8,959,756,939 bytes，传输后逐文件 SHA256 核验。该快照中的历史结果不替代 `hss/` 内的最终实验结果。

失败、取消和探索性实验也保留原标记；**保存下来不意味着实验有效或全部结论成立。** 旧 steering 独立 AI 逐题语义审核仅完成 **43/202**；用户后续人工汇总与逐题审核文件分开保存，不能合并宣称 202 题均已独立审核。

## 2. 核验记录

总清单 `manifest.json` 的 SHA256：

```text
9c13f7e2a5c4110cf4778c3dfa42f88f37088662d93e28c0dc8b000735ed050b
```

- 上传时计算每个文件 SHA256，并核对已有 OpenAct publisher 哈希；每个 multipart 核对 MD5/ETag，每个完整对象核对长度和复合 ETag。
- 每包首、中、末非空文件均实际 Range GET 回读并核对 SHA256。最终重新检查冻结源文件的大小和修改时间，再验证全部云对象，才发布总 `_SUCCESS.json`。
- 独立完成复核重新读取云端总清单和完成标记，确认与 Lambda 副本逐字节一致；重新核对 **全部 185 个云对象 HEAD 和云端收据**，总文件数和字节数一致，无错误。
- 已有 MATH/MMLU 的云端完成标记也重新读回，与引用记录一致。本次没有重新下载其全部 2.21 TB，也没有二次完整下载本次 776 GB 存档。
- `_SUCCESS.json` 的 `at` 字段保留的是最终复核开始时间 14:40:47 UTC；实际完成标记发布时间以云端 `LastModified` **15:01:22 UTC** 为准，独立审核文件已说明。

Lambda 状态/收据目录：`/lambda/nfs/dami/research-archive-20260926/`。

本地轻量清单、185 个收据及下载核验记录：`results/research-archive-20260926/`。独立完成审核在 `delivery/independent_archive_review.json`；逐包恢复索引摘要在 `delivery/archive_receipt_summary.json`。最终说明及追加报告通过 R2 `delivery/` 保存，不修改已冻结的归档对象。

## 3. Reliability 的实际结论

Qwen2 × MATH 5,000 条、29 层 raw response token mean 的完整对角 GMM 可靠性实验已审核完成：29,783 个候选、89,349 次初始化全部收敛，5,278 条选择记录、377 条固定 K 记录、1,334 次中心匹配。没有严格最优 K 命中 80。

- 五种子下，2% ICL 容差使逐层 K 的平均绝对偏差从 **3.9669 降至 1.1034**。
- 匹配中心平均余弦距离：种子 **0.008753**，子样本 **0.013691**；成员一致性 ARI 为 **0.5742 / 0.4891**。中心较稳定，成员划分没有近乎完全一致。
- **29/29 层**平均簇间 KL 大于簇内拟合 KL。
- K 对样本量明显敏感：index 14 的 20% / 50% / 90% / 100% 数据分别选 **5 / 17 / 29 / 33**。不能写成样本量不影响簇数，也没有统一的 L10 峰值。

完整方法、数值与限制见 [reliability 最终报告](qwen-math-reliability-final-20260926.zh-CN.md)。没有为匹配论文预期而修改协议。

## 4. 恢复步骤

1. 用具有该 R2 bucket 访问权的客户端下载新存档根下的 `_SUCCESS.json`、`manifest.json`、`receipts/` 和 `delivery/archive_receipt_summary.json`。核对清单 SHA256 与上值一致，完成标记记录 185 包、1,164,373 条目。
2. 按清单下载 `archives/*.tar`。**解压前**核对每包大小和 `receipts/<包名>.json` 中的完整 tar SHA256；ETag 不是 SHA256。
3. 先检查清单列出的符号链接及 tar 路径，再将全部包解压到一个空目录，保留 `hss/`、`openact-runs/`、`hss-input-cache/`、`local-reports/`、`code/` 的相对结构。符号链接目标内容没有被自动包含。
4. 只需要个别文件时，下载 `indices/<包名>.json`，使用记录的 offset/bytes 做 Range GET，并核对该文件 SHA256。不要把某个包的索引当成所有实验文件的总索引。
5. 用 `code/hss.bundle`、`code/openact.bundle` 恢复 Git 历史，或展开同目录源码 tar。按各实验冻结协议重建依赖、固定模型/数据 revision，并调整机器相关绝对路径。环境和模型权重需另行安装/下载。
6. MATH/MMLU 原始采集从旧 R2 前缀单独恢复；其完成收据随新存档根的 `previous_math_mmlu_SUCCESS.json` 一并提供。

Git bundle 的冻结版本为 HSS `ddb6945`、OpenAct `9fdc490`，含 GSM8K 分支 `1167d54`；后续最终报告和交付文档在 GitHub 当前研究分支及 R2 `delivery/` 补充保存。HSS 分支为 `codex/tokenmean-replacement-20260924`，没有自动合并或改写历史。

## 5. 后续状态

采集、拟合、steering 和归档任务均结束，不自动重启。半小时监控在交付后恢复为每天纽约时间 21:00 的 Git 检查，仅提交经过验证的本任务代码/文档；不提交数据、模型、凭据、无关 WIP，不制造空提交。
