# A 成员 11 月 6—8 日交付对照

## 11 月 6 日：端到端测试与问题排查
- A 侧完成统一编排入口、B/D 适配、C 最新代码接入。
- 完成六子任务依赖状态检查、CNP 两份配置 Bid 评分授予、只读任务不推进环境、B 控制任务经过 D 双重审核。
- 完成本地合同测试；真实 SMPL 运行仍需在安装了 B 依赖的团队设备执行。

## 11 月 7 日：架构、接口、运行说明与修复
- 已提供 `ARCHITECTURE.md`、`INTERFACE_CONTRACT.md`、`REPRODUCE.md`、`KNOWN_LIMITATIONS.md`。
- 保留 B 真实 PASS/REJECT 日志和 C 的合成数据对照实验原始数据。
- 所有证据按来源区分，不将 C 合成 Bid 说成真实投标，不将 FakeAdapter 测试说成 SMPL 实验。

## 11 月 8 日：冻结前检查
- 代码入口、依赖锁文件、SMPL vendor 源码、示例配置、测试和运行脚本均随包提供。
- 必须在团队演示设备运行 `scripts/run_acceptance.ps1`，生成本次 A 主程序真实 `A/B/D` 日志。
- 若现场运行失败，保存完整 traceback 和环境信息，状态标记为受阻，不得宣称端到端验收通过。
