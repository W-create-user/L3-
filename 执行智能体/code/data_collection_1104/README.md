# 成员 B 11 月 4 日数据采集

本模块完成当日任务：保存观测、动作、奖励、终止状态及时间步；输出 CSV/JSON；记录每次实验的初始条件、动作序列、模型版本和依赖。基于当前 integration_abd_1103 接口，调用 A 已提供的 DecisionAgent/HTN 代码和 D PR #1 的固定源码，在真实 SMPL 中连续运行。

## 一 本机运行

```powershell
cd D:\SMPL_Project\member_b_1101
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_B_1104.ps1
```

默认每轮执行 5 个合法动作。正常轮直接运行；另一轮在第二步后插入一次越界候选，由真实 D 拒绝后继续。两轮各 reset 一次，轮内连续操作同一个环境，不在每个动作前 reset。

每次创建 data_1104/runs/日期时间_唯一编号，不覆盖已有记录。英文运行目录用于避免 CasADi 的中文路径 DLL 问题。复制到 D:\大创 的 CSV 和报告可直接阅读，模型仍在英文项目目录运行。

可修改采集步数与种子，例如 -Steps 3 -Seed 1101。代码允许 1 到 60 步；遇到环境终止、错误奖励或异常会保存数据并停止，不能据此保证任意时长稳定控制。

## 二 每轮保存什么

| 文件 | 内容 |
|---|---|
| manifest.json（运行根目录） | Python、依赖、SMPL 提交号及实际文件哈希、A/D/B 版本、模型选项、边界、种子和输入快照哈希 |
| inputs/atropineenv | 本次使用的官方 x0.txt、z0.txt、model.npy 原样副本 |
| episodes/实验名/initial_conditions.json | 实际初始观测、计数器、seed，以及内部 xk/zk/xe 数值快照 |
| steps.csv | 每个候选一行：任务/请求 ID、动作各通道、审查、执行状态、奖励、done、计数和时间 |
| observations.csv | reset 一行，随后每次有实际执行返回时增加一行，观测逐分量保存为数值列 |
| events.jsonl | 完整 Task、ActionRequest、审查、执行反馈与前后状态等原始结构化事件 |
| action_sequence.json | 候选动作顺序，包含被拒绝候选，并明确 executed=false |
| B_events.jsonl | B 接口原始事件 |
| D_raw_reviews.jsonl | D 原始审查日志 |
| main_attempt_*.jsonl | A 模板各阶段及其前置条件、执行和反馈日志 |
| episode_summary.json | 本轮实际次数和结束原因 |
| experiment_index.csv | 两轮实验汇总 |
| verification.json | CSV/JSONL 一致性校验和实际采集结果 |

内部状态快照是仿真器内部数据，不当作真实传感器观测。复现方法是使用相同源码、配置、依赖和 seed 调用 reset，不能把 39 维观测直接当作 SMPL 的完整 initial_state。

## 三 CSV 字段怎么读

steps.csv 的 action_00 到 action_03 是四个候选动作偏移；reward 为该次真实模型返回值；拒绝没有模型奖励，单元格留空，不伪造为 0。execution_status、approved、executed、success 区分审查与模型执行。失败原因保留原文。

step_count_before/after 是当前轮模型步数；step_calls_before/after 是实际 env.step 调用累计次数。被拒绝时两类计数和 state_version 均不变。

nominal_sim_time_*_min 根据实际 step_count × 模型 dt 计算，本次 dt=10 分钟。它是模型时间，不能解释为电脑实际运行时长。wall_duration_ms 是包含任务处理、审查、模型调用和主程序事件写入的调用耗时，不是纯求解器耗时。

observations.csv 的 obs_000 到 obs_038 对应当前环境的 39 个观测分量。obs_000 是上游自增前的步号，首次执行后仍可能为 0，所以数据表使用额外 step_count 对齐。其余含义见此前给 D 的 observation_schema.json。

CSV 使用 UTF-8 BOM，可直接用 Excel 打开。原始JSONL采用标准JSON；诊断中的非有限数使用 __nonfinite__ 标记，不用非法 NaN 字面值，也不改成 0。

## 四 数据复查

```powershell
.\.venv\Scripts\python.exe -m data_collection_1104.validate_data "D:\SMPL_Project\member_b_1101\data_1104\runs\具体运行编号"
```

校验程序重新读取已保存的 CSV 和原始 JSONL，检查任务编号、请求编号、动作、奖励、步数、模型时间和观测是否一致，并检查被拒绝动作没有生成额外模型状态行。

8 项记录测试覆盖正常与拒绝、零奖励、重复记录、计数不一致、CSV 被改动、初始数据保存、非有限诊断值和异常记录。快速测试使用手工构造数据；真实模型证据保存在运行目录。

## 五 结果范围

动作仍使用 B 的中点接口基线，没有训练控制器。第二轮插入拒绝候选用于验证记录和门控，不是控制策略性能对照。当前调用 A 压缩包 member_A_11_02 中的真实旧版代码，未宣称取得最新版生成器；D 使用实际下载的监管源码。C 调度、R-03 和长期优化不在本次实现中。

原有接口代码不需要修改。A 后续集成可使用 ExperimentRecorder，在其调用前后传入快照和执行结果，按实际运行保持同样的日志格式。
