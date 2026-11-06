# 成员 C 阶段工作报告（截至 2026-11-06）

## 1. 工作范围
依据项目《具体工作.docx》《具体工作安排_修订稿.docx》，本报告覆盖 C 负责的 SchedulingAgent/CNP：任务注册、依赖管理、公告投标评分授予、执行反馈、失败回退/重分配、固定策略与 CNP 选择策略对照实验、接口与验收材料。

## 2. 已实现功能
- 以 `subtask_id` 为唯一调度键，保留父 `task_id`。
- HTN 子任务批量注册；按 dependencies 和 priority 生成 ready_tasks。
- CNP 公告、Bid 字段校验、重复投标拦截、加权评分、Assignment。
- `running/completed/failed` 反馈；失败原因保留；可选 `exclude_failed_agent`。
- 失败重分配、活动执行者锁、Assignment 历史和查询。
- 无有效 Bid 返回 `unassigned` 并保持 `announced`，支持调用方发起下一轮收集。

## 3. 评分规则
`score = 0.50 * capability_match + 0.30 * (1 / (1 + estimated_cost)) + 0.20 * (1 - load)`。
权重为人工设定的可解释原型参数，不是训练或优化得到的最优权重。

## 4. 自动化测试
当前测试命令：`python -m unittest discover -s tests -v`。
测试结果：16 tests，全部通过。测试使用本地 Python 和合成 Bid，不等于 B 的真实 ExecutionAgent 联调。

## 5. 固定策略与 CNP 对照实验
实验脚本：`experiments/run_fixed_vs_cnp.py`。
样本设计：30 个确定性合成任务；每个任务 3 个候选执行者；两个策略使用相同候选集合。
- 固定基线：按 agent_id 字典序选择第一个候选者，不使用 Bid 指标。
- CNP：按上述加权评分排序。
数据文件：`experiments/results/fixed_vs_cnp_raw.csv`；汇总：`experiments/results/fixed_vs_cnp_summary.json`。

| 指标 | 固定 first-fit | CNP 加权评分 |
|---|---:|---:|
| 任务数 | 30 | 30 |
| 分配成功数 | 30 | 30 |
| 平均能力匹配度 | 0.7870 | 0.8523 |
| 平均预计成本 | 4.0000 | 3.2000 |
| 平均负载 | 0.4150 | 0.3057 |

### 解释边界
以上是可复现的算法级合成数据实验，不是 B 的真实 Bid，不是 SMPL 仿真结果，也不证明端到端任务完成率提升。两种策略的分配成功率均为 100%，因为每轮均提供了 3 个有效候选者；该指标没有区分能力，不能用来宣称 CNP 胜出。实验仅显示在这组构造数据中，CNP 选择的候选者平均能力匹配度更高、估计成本更低、负载更低。仍需用 B 的真实投标数据复验。

## 6. 未完成/外部依赖
- B 的真实 Bid 和 ExecutionResult 尚未纳入本包实验。
- C→B Assignment 接收和 B→C 反馈的完整 main.py 运行日志尚未提供。
- 真实投标超时计时/异步收集不在 C 当前实现内；A/main.py 负责检测超时并启动下一轮。
- 与 SMPL 的真实执行效果、奖励、环境步数不由 C 单独验证。
- 11 月 8 日现场冻结前，仍需团队在集成仓库执行完整主流程并保存日志。

## 7. 11 月 8 日 C 验收清单
- [x] CNP 代码与接口文档整理
- [x] 失败恢复与重分配本地测试
- [x] 固定策略 vs CNP 可复现实验脚本
- [x] CSV 原始合成实验数据及 JSON 汇总
- [x] 评分公式、数据来源、局限性说明
- [ ] 集成仓库现场运行 main.py 并留存日志（需 A/B/D）
- [ ] 用真实 B Bid 重跑固定策略 vs CNP 对照实验
- [ ] 核对报告中的所有系统级结论与原始运行证据

**结论：C 的代码整理、本地测试和算法级对照实验材料已形成；系统级验收仍依赖 A/B/D 的真实集成运行，不能在 C 单模块环境中代替完成。**
