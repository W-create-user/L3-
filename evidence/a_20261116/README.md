# A 2026-11-16 真实 Atropine 运行证据

本证据包经用户于2026-11-16明确授权公开上传。对应代码提交 `37ef27173d5b986001ebf95065020feb251100fd`。

文件：[real_atropine_runs.zip](real_atropine_runs.zip)

- 大小：137834 bytes
- SHA256：`ea38c00a1205e7bb5bb03e3274ae35cb8c307290696999df95ffd45160b2c5c8`
- `validation/`：三种子正常/拒绝/只读，以及连续五轮运行的原始结果、B/D日志与事件；汇总见包内 `validation/summary.json`。
- `session/`：候选替换、重复拒绝、接管阻断与显式恢复的原始结果和事件；共2次真实step，汇总见包内 `session/result.json`。

范围：验证既有物理链和恢复逻辑。新增数值目标 HTN 没有在这些运行中驱动模型；这些结果不证明生产目标达成、R-03工艺安全、mAb验收或生产效率提升。实际运行依赖版本在包内汇总列明，SciPy 1.17.0与仓库1.16.3锁定版本不同。
