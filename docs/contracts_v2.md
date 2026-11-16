# 四智能体接口 v2 草案

2026-11-12，A01/A02 当日交付。版本 `2.0-draft`，状态为供 B/C/D 审阅，未完成团队冻结。本文件区分当前实现与新增提案。当前 main 基线为 `2e88c200afa4698b0dc13a9d2260fbaf44d56972`。

## 当前实现与迁移边界

现有 DecisionAgent 仍接收两个有限目标类型，Task 经 a_to_c_adapter 转给 C；B 的 normalize_task 接受原 Task 或三字段表示；D 审核和 B 执行门控继续使用原请求。此次不改变这些调用，不把旧对象自动标为 v2，也不修改 main/runtime 的控制逻辑。

本次只有 `l3_system/planning/goals.py` 的 GoalSpec 和规范化变量注册表校验可执行。下列消息信封为设计草案，尚无传输层、全消息校验器或主链迁移。后续 A05/C04 实现并测试后才能将其标为运行接口。

## 共同消息信封提案

每条消息以共同头部加 `payload` 表示，业务对象的旧字段保持在 payload 内。不得由调用方随意改写执行端身份。

| 字段 | 类型与语义 | 产生方与一致性要求 |
|---|---|---|
| schema_version | 固定字符串 `2.0-draft` | 发送与接收共同协商；未知版本拒绝 |
| message_type | GoalSpec / Task / Assignment / ActionRequest / SafetyDecision / Feedback | 与 payload 类型一致 |
| environment_id | 非空字符串，标识具体环境配置 | B 给出含后端、版本、归一化/坐标配置的登记身份；Atropine 与 mAb 不共用 |
| correlation_id | 一次目标会话的不可变关联 ID | A 创建，所有阶段回传；它不是幂等键 |
| episode_id | 一个环境 reset 周期的 ID | B 创建，reset 必须更换；跨 episode 请求拒绝 |
| payload | 对应消息对象 | 与下表定义一致 |

环境身份具体编码等待 B 确认，不能用不含配置的环境简称消除归一化差异。D 当前 process_rules 使用 `environment`，拟由显式适配器映射到 environment_id；映射必须验证等价，不能仅改字段名。D 的 `coordinate` 用词沿用，避免额外引入 scale 同义字段。

## 六类对象的字段与所有权

| 对象 | 生产者 → 消费者 | 当前或拟保留的核心字段 | 新增/待确认内容 |
|---|---|---|---|
| GoalSpec | 用户配置经 A 校验 → A | 旧入口仅 task_type、description、priority 等；它不是生产 GoalSpec | 本节后详列数值目标；所有字段均为新增草案 |
| Task | A → C/B | task_id、task_type、priority、preconditions、dependencies、status、completion_criteria、description、constraints、subtasks、source | 生产任务拟携带 goal_id；重规划的 parent_plan_id 由 A04 定义，不改变已有 ID |
| Assignment | C → 被授予角色 | task_id、subtask_id、winner_id、assignment_status；score 由竞价结果提供 | 独立 assignment_id、资源租约及超时由 C03 确认；调度授予不等于安全放行 |
| ActionRequest | B → D → B | request_id、task_id、subtask_id、state_version、action；B 内部保留待执行快照 | 从所属 Assignment 继承关联头；state_version 必须来自 B，准备动作不能 step |
| SafetyDecision | D → B/A | request_id、task_id、subtask_id、decision、rule_id、reason、risk_level、demo_mode | v2 拟回显 state_version 和全部关联头；decision 仅 pass/reject，不从风险分数猜测放行 |
| Feedback | 各阶段 → C/A | task_id、subtask_id、execution_status、failure_reason、exclude_failed_agent | B 原始结果另外保留 request_id、是否 executed、前后 state_version 与步进计数；非物理阶段 request_id 明确为 null |

当前 D→B 适配器返回字段未包含 state_version，当前 B→C feedback 也不保留全部执行证据；因此这两处 v2 迁移属于后续工作，不得声称新头部已全链传播。原始 B 结果与事件日志必须继续保存。

状态词不可混用：Task 有 created/pending/running/completed/failed/blocked；B 的 rejected 经现有反馈适配器映射为 C 的 failed，同时保留原始 safety_decision。v2 需要回传原始执行状态，不能把安全拒绝统计为执行成功。GoalEvaluation 的 succeeded 与任务链 completed 分开统计。

## 关联与重放约束提案

1. B 唯一推进环境；A/C/D 的准备、读取、评估不 step。
2. task_id/subtask_id/request_id/state_version 从源对象传递，接收方不得生成替代值补齐错误消息。
3. SafetyDecision 必须关联完全相同的环境、episode、任务、子任务、请求、状态版本和候选动作。审核通过后状态改变必须重新准备和审核。
4. request_id 的一次性消费仍由 B 保证；correlation_id 用于追踪，不能代替 request_id 的去重语义。相同 correlation_id 可有多个合法请求。
5. 未知执行结果保持待对账或接管，不能因通信超时重放 step。C 的释放/重授予与 B 的执行事实须对账。
6. JSON 不允许 NaN/Inf，时间值必须携带时钟/单位语义。仿真时间、墙钟、step_index 不互换。安全上下文由可信执行端/provider 提供，不能由前端自报后绕过审核。

这些是 v2 验收要求，不等于现有本地消息已实现所有跨进程保证。

## GoalSpec 首版及已实现校验

| 字段 | 规则 |
|---|---|
| schema_version | `2.0-draft` |
| goal_id | 非空且无首尾空格 |
| environment_id / registry_version | 必须与可信注册表精确一致 |
| variable | 注册表中的 observation 变量名；不得用 action、reward 或未知字段代替 |
| unit / coordinate | 与变量声明精确相等；不隐式换算、不裁剪 |
| target_range | lower/upper 两个有限 JSON 数值，闭区间且 lower ≤ upper；若已声明变量数值域，则整个目标区间须在域内 |
| max_steps | 正整数，拟定义为从目标启动开始允许的成功物理步进预算 |
| hold_steps | 正整数且不大于 max_steps；拟要求连续不同步号的执行后观测都达标 |
| allowed_actions | 覆盖全部动作轴，每项有 unit、coordinate、range；范围必须是 B 模型动作域的子集 |

注册表必需字段：schema_version、environment_id、registry_version、evidence_ref、observation_size、action_size、variables。每个变量有 source、index、unit、coordinate、bounds；source/index 唯一、索引不越界，动作轴必须全部登记。动作 bounds 必填，观测 bounds 可以 null；null 不推断任何物理限值。

schema/单位/来源引用只是声明校验；代码不访问 evidence_ref，也不证明数据确实按声明坐标转换。可信注册表必须由 B/D 审阅后从系统配置提供，不能接受不可信请求自行携带一个“相符”的字典。

`unconfirmed/unknown/pending/TBD` 元数据拒绝。仍需人工核实其他自由文本证据。缺失或多余字段、布尔值冒充整数、数字字符串、NaN/Inf 均拒绝；返回深复制，修改返回结果不修改输入。

## 目标评价的下一步契约（未实现）

拟实现 `evaluate_goal(observation, history, goal)`：返回 evaluation_status（in_progress/succeeded/failed/unevaluable）、in_range、deviation_to_range、consecutive_steps、remaining_steps、reason。区间内偏差为 0，区间外返回到最近边界的距离，单位同目标。

保持窗口仅计同 episode、同目标、严格连续且不重复的物理步进后观测；reset、缺失步、无效观测不得跨越拼接。最后一个允许步若满足完整窗口可成功；预算用尽仍不满足则 failed。缺映射/单位冲突/非有限数值为 unevaluable，绝不成功。提前终止未满足窗口的原因应保留，不能用 reward 或模板步骤完成替代工艺达标。

13 日实现并测试上述语义；14 日提交团队契约复核；16 日接 B03/D02；18 日真实目标验证。本次未实现评价器、历史记录器或其主链集成。

## B/D 材料接收及待确认项

B 材料固定提交 `c3dc5cc12d5f5bdd2ae39673bcf74529a24912d3`，目录 handoffs/B_to_D_20261112；D 固定提交 `b966f8af63cf3cf2370b10eec752501219da1a71`。

- B 已提供 Atropine 39 维字段顺序与来源；第 0 项是自增前步号，不是物理分钟。4 路动作是 delta_from_USS，绝对物理流量需加 USS；不能拿动作偏差当绝对流量。
- B 的 mAb 单步探测不是跨工序目标验证；多项单位仍未确认，不创建可运行 mAb GoalSpec。
- D 已提供可配置围栏和合成数据统计检测框架；真实工艺阈值仍待确认。GoalSpec 的允许动作范围只能收紧 B 的范围，不能替代 D 审核。
- 不把 observation Box 当工艺安全限值；不为赶进度填入生产目标值。configs/goals 下只有明确标识的合成夹具，没有启用的 Atropine/mAb 目标。

| 复核方 | 必须确认的内容 | 截至本次状态 |
|---|---|---|
| B | environment_id、episode/state_version 生命周期、观测名称单位坐标、时间、动作绝对值/偏差 | 已读交付，未取得对本 v2 草案的确认 |
| C | correlation/assignment 关联、状态词、租约及迟到反馈、版本迁移 | 待确认 |
| D | 目标与安全围栏区别、规则上下文、恢复条件、单位坐标 | 已读交付，未取得对本 v2 草案的确认 |
| 导师 | TEP/SMPL、算法及通信路线差异 | 待确认，详见 requirements_november.md |

## 本地验证

仓库根目录运行 `python -m unittest l3_system.tests.test_goal_spec -v`。合成夹具只证明输入契约，不能用作过程安全限值或生产目标达成数据。现有运行入口不消费 GoalSpec；显式传入它仍不能作为可执行生产目标运行。
