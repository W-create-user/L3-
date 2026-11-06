# L3 系统架构与 A 侧集成关系

```mermaid
flowchart TD
  U[生产/控制目标] --> A[DecisionAgent]
  A --> H[HTN 六个有序子任务]
  H --> C[SchedulingAgent / CNP]
  C -->|公告、两份配置型 Bid、评分、授予| X{winner_id 校验}
  X -->|A 逻辑任务| AH[A 侧任务处理器]
  X -->|D 监管任务| D[MonitoringAgent v1.2]
  X -->|B 执行任务| B[ExecutionAgent]
  AH --> F[C update_feedback]
  D -->|预审结果| F
  B -->|ActionRequest| D
  D -->|明确 pass| G[EnvironmentAdapter]
  D -->|reject / 异常| R[阻断，不调用 step]
  G -->|ExecutionResult| F
  F --> C
  C -->|依赖解除/状态| A
  A --> P[ReplanningController]
  P -->|有限动作重生成/失败重分配决策| C
```

## 关键边界
- AtropineEnv 是当前短期工程验证环境；mAbEnv 是研究方案中的目标/扩展模型，不能把本次 AtropineEnv 结果写成 mAbEnv 验证。
- D 的 R-01/R-02 在执行前检查动作格式、数值和真实 action_space 边界；R-03 未启用。
- HTN 的 `safety_review` 阶段会执行一次预审；B `execute_assignment()` 内部还会再次调用 D，第二次审核是紧邻 `env.step()` 的权威门控。
- 本项目 CNP Bid 是本地配置的能力档案 Bid；不是来自多个独立在线执行器的实时遥测。
- B 当前 `space_midpoint` 只是接口基线策略，不是经优化或训练得到的过程控制器。
