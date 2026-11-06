# 已知限制与真实性边界

1. 当前运行容器未安装 `gym==0.26.2`，且安装源 DNS 不可用，因此本次无法在此容器复跑真实 AtropineEnv。B 提供的 2026-11-06 PASS/REJECT 日志保留为成员实测证据，但不能写成 A 新主程序已完成 SMPL 端到端运行。
2. 本地完整六子任务集成测试使用真实 B `ExecutionAgent` 类和真实 D `MonitoringAgent`，但使用 FakeAdapter 替代 SMPL 环境。其用途是接口/状态机合同测试，不是环境实验。
3. CNP 测试 Bid 由 `configured_bids()` 根据本地能力档案生成，明确标记为 `configured_local_capability_profiles`；没有 B 的动态 Bid 采集接口，不能称为多个独立执行器的实时投标。
4. `space_midpoint` 是 B 提供的本地接口基线，不是经过过程目标优化、强化学习或模型预测控制训练的策略。
5. D R-03 未实现/未启用。当前只接入 R-01 输入合法性和 R-02 动作边界检查。
6. 本次六子任务任务模板是项目自定义 HTN 式有限模板，不是通用 HTN 求解器。
7. AtropineEnv 是本轮先行工程验证对象；项目方案建议 mAbEnv 作为后续目标/扩展模型。不得声称已完成 mAbEnv 验证或覆盖完整工厂所有外围环节。
8. 仿真验证不等同真实工厂现场验证，不构成功能安全认证或工业安全联锁验证。
