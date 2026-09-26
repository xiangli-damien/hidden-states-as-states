# HSS / OpenAct 完整存档（2026-09-26）

用户授权：完成当前 Qwen MATH reliability，并把实验资料存档、上传 Cloudflare R2 和 GitHub，全部完成后通知。

## 范围

- 先完成既有 reliability 的全部 29,783 个候选、固定 K 重拟合、中心比较和最终审核。科学协议不变，采集与旧实验不重启。
- R2 保存 `/lambda/nfs/dami/hss` 全部实验文件、OpenAct 后续采集和历史辅助运行、`/home/ubuntu/hss-cache/data` 拟合输入缓存，以及两个仓库的 Git 快照和历史 bundle。失败/取消实验仍保留原始标记，归档不意味着其结果有效。
- 已上传的 MATH/MMLU 原始数据（57,126 条、573 分片、约 2.21 TB）引用原 R2 存档；本次验证旧 `_SUCCESS.json` 的协议、分片和样本数，不重复传输，不声称重新下载核验全部 2.21 TB。
- GitHub 保存本任务代码、配置、复现说明和轻量结论。大型数据、模型拟合文件和凭据不提交 Git。作者和提交者均为 Xiang Li 的已绑定邮箱。保留无关 WIP，不自动合并、不 force push。

## 位置

- 新 R2：`s3://autoact-data/hss/research-archive-20260926/`
- 已有原始数据：`s3://autoact-data/openact/math-mmlu-full-20260916/`
- 上传状态：`/lambda/nfs/dami/research-archive-20260926/`
- 入口：`scripts/archive_research_r2.py`；配置：`configs/research_archive_20260926.json`。

## 完整性和恢复

等 reliability 的 `audit.json`、`COMPLETE.json` 及进程退出全部成立，再冻结源文件清单。每约 4 GiB 原文件划为一包，以流式 tar 从 Lambda 直接上传 R2，不经个人电脑、不占用本地 SSD 存放大型 tar。每文件记录 SHA256，逐 multipart 验证 Content-MD5/ETag，完整对象核对复合 ETag 和大小，另对每包首/中/末非空文件做实际 Range GET SHA256 验证。没有声称二次完整下载每个包。

上传失败保留原始数据、已接受的 multipart 和日志，允许断点恢复；已有不同对象不覆盖。最终再次检查源文件大小/时间戳和全部云对象，全部通过才发布 `_SUCCESS.json`。`manifest.json`、`indices/` 和 `receipts/` 是恢复和审计入口。符号链接不跟随，目标在清单中明确记录；运行环境和模型下载缓存不包含在数据归档中。

恢复时先核对存档 SHA256，再把全部包解压到同一个空目录，得到 `hss/`、`openact-runs/`、`hss-input-cache/`、`code/`。代码从 bundle 克隆或从源码 tar 提取；根据各实验冻结协议安装环境并映射旧绝对路径。先检查符号链接目标。MATH/MMLU 从原存档恢复。

本任务只有在科学审核、R2 全部验证、GitHub 提交均完成后才算交付完成；仅启动上传不算完成。现有半小时自动任务继续跟踪到交付，再恢复每天纽约时间 21:00 的 Git 检查。
