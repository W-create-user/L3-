# 11 月 8 日验收矩阵（状态按当前证据填写）

| 验收项 | 当前状态 | 证据/说明 |
|---|---|---|
| A 能输入目标并输出结构化 Task | 已有代码/本地测试 | `decision_agent.py`、`tests/test_a_integration.py` |
| HTN 分解多个有序子任务 | 已有代码/本地测试 | `control_cycle` 模板六个子任务及依赖 |
| A→C 注册、依赖阻塞与解除 | 已有本地验证 | A/C 自动化测试；闭环测试使用测试环境 |
| CNP 至少两个候选者投标并评分授予 | 已有算法级/合成数据证据 | C 的 30 组确定性合成 Bid 对照实验；不是实时 Bid |
| C 执行失败记录和失败重分配 | 已有本地测试 | C 16 项测试，含失败执行者排除、历史、资源锁 |
| B 报告的 59 项回归测试 | 成员交付报告 PASS | `evidence/B_20261106/regression_tests_59/verification.json`；测试源码未随清单完整提供，本次未独立复跑这 59 项。 |
| D 在真实动作空间上 R-01/R-02 | B 提供真实运行证据 | `evidence/B_20261106/logs/.../verification.json`；D demo_mode=false |
| B 合法动作经 D PASS 后执行 SMPL | B 提供真实运行证据 | B PASS 日志，step_delta=1 |
| B 越界动作经 D REJECT 不推进 SMPL | B 提供真实运行证据 | B REJECT 日志，step_delta=0、R-02 |
| A/C/B/D 六子任务统一编排代码 | 已完成本地合同测试 | `main.py`、`tests/test_live_orchestration_contracts.py` |
| 六子任务统一编排（实际 B Agent + 实际 D，测试 Adapter） | 本地通过 | 34 项总测试中的 3 项；明确不是 SMPL 真实运行 |
| A 新主程序直接连接真实 SMPL 的端到端六子任务 | 待在团队设备运行 | 当前执行容器缺 Gym 依赖且无法联网安装；请用 `scripts/run_acceptance.ps1` 在 B 已配置环境运行 |
| 报告与原始证据逐项对应 | 部分完成 | 本目录保留 B 真实日志、C 合成数据和测试状态说明；现场日志待补 |
| R-03 | 未完成/不纳入本轮验收 | D 明确仍在研究阶段，未接入主程序 |

**总原则：** B 的真实 PASS/REJECT 证据只证明 B+D+SMPL 的单控制任务安全执行链；不能替代 A+C 真实调度与六子任务端到端运行证据。
