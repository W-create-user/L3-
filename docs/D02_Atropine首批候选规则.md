# D02 Atropine 首批候选规则

2026年11月12日草案。本次将B交接中的字段转成可核对的首批候选规则，所有工艺限值和运行预算阈值保持null，所有候选enabled=false。本表不是已启用规则库，也不证明数据已达到安全或科研指标。

固定来源：仿真源码提交 `2e88c200afa4698b0dc13a9d2260fbaf44d56972`；B交接提交 `c3dc5cc12d5f5bdd2ae39673bcf74529a24912d3`；D规则引擎所在提交 `8457fe7a847b758a019dc5d824b520c5f434e9ac`。本表仅适用于B已说明的Atropine 39维观测、4维动作、normalize=false、uss_subtracted=true。mAb 1970/9不适用。

## 先保留已确认的软件约束

| 约束 | 确认范围 | 不能据此推断的结论 |
|---|---|---|
| R-01 | 动作缺失、不能转换成有限数值数组或形状不符时拒绝 | 不能宣称D函数拒绝全部数字字符串或布尔值；B另有dtype校验 |
| R-02 | 以live action_space的low/high做包含边界的检查 | Box不是现场或工艺安全认证，不作为R-03阈值 |
| STATE-INVALID | 数值、有限性与配置的观测形状检查 | 数据完整不表示工艺正常；check_action的形状校验依赖observation_space |
| B执行与会话条件 | ready、请求state_version匹配、未结束episode、未知执行结果不得盲重试 | 这些软件条件不是虚构的工艺ready传感器，也不替代工艺联锁 |

## 首批八项候选

下表的阈值均待有依据地制定。机器可读版本为 [atropine_rule_candidates.draft.json](../configs/research/atropine_rule_candidates.draft.json)，每项含来源路径、索引、单位可信度、待补输入和负责成员。

| rule_id | 输入 | 候选检查 | 阈值 | 成立前提 |
|---|---|---|---|---|
| R-03-EFACTOR | obs[6] E_factor_current，无量纲 | 允许工况内的状态范围 | lower/upper=null | A02定义目标与允许偏差；B/D核验过程依据 |
| R-03-Q-ORGANIC | obs[37] Q_organic，B标注L/min | 有机相流量范围 | lower/upper=null | B确认模型适用范围及过程依据 |
| R-03-Q-AQUEOUS | obs[38] Q_aqueous，B标注L/min | 水相流量范围 | lower/upper=null | B确认模型适用范围及过程依据 |
| R-03-RATE-A0 | action[0]，USS增量坐标 | 与上次已执行动作的变化率 | max_rate_per_second=null | B确认单位、初始控制量、时间和限速依据 |
| R-03-RATE-A1 | action[1]，USS增量坐标 | 同上 | max_rate_per_second=null | 同上 |
| R-03-RATE-A2 | action[2]，USS增量坐标 | 同上 | max_rate_per_second=null | 同上 |
| R-03-RATE-A3 | action[3]，USS增量坐标 | 同上 | max_rate_per_second=null | 同上 |
| R-03-FRESHNESS | 可信上下文中的审核时间与观测时间 | 同一时钟域内的数据年龄 | max_age_seconds=null | A/B/C确定采集和传输预算；D验证缺失/过期/未来时间 |

E-factor规则拟表达允许工况或目标围栏，不自动表示化学危险。流量规则拟表达操作域，不能从“非负”或Box范围直接推成工艺限值。是否仅设上限、下限，或需要随工况变化，均须在依据形成后调整规则版本；当前range类型只是候选接口形态。

动作变化率拟按 `abs(candidate_delta - last_executed_delta) / elapsed_seconds` 计算。B声明的动作单位仍带源码换算推断性质，率的单位也必须保留该限定。只有USS固定时，增量变化才与相应绝对输入变化一致；若后续USS变化，应同时记录USS并重新确定坐标约定。首次动作没有可靠历史控制量时不能填零假装已知，需B给出初始化控制及其生效时间。被拒绝的候选不是执行历史。

观测新鲜度用采集与审核时间；B的10分钟仿真采样周期不能直接变成600秒的墙钟允许延迟。跨进程时钟身份、同步和网络延迟应由A/B/C的通信契约说明。

## 未选入首批的字段

obs[0]是自增前步号，不是物理分钟；obs[1:5]是USS基准，不是当前动作；obs[5]是稳态E-factor参考。obs[7:9]的两个估计器坐标未有已确认的现场物理对应，先留作分析。obs[9:37]的组分流量继续保留原始记录，待A02明确生产目标和独立依据后再决定是否纳入，不从统计响应排名直接得出危险等级。

本次没有凭空增加前置工艺联锁：当前字典没有被确认的“工艺ready”离散传感器。若要新建precondition规则，先确定实际变量、意义、取值和解除条件。已有执行ready与资源锁条件仍由原软件链检查。

## 裁剪与信息损失

源码 `vendor/smpl/smpl/envs/atropineenv.py` 的 `_step` 在返回前执行 `observation.clip(min_observations,max_observations)`。因此超界的真实模型值可能被截在边界，返回值未越界不能证明工艺未越界。B提供的是返回观测，D不能从中复原未保存的超界幅度。

采集时应尽可能另外记录裁剪前向量、饱和标记或可直接取得的过程info，并说明可取得的范围。新增记录口由B通过唯一执行通道提供；D不为补日志额外调用step。无法取得时明确标为不可用，并把结论限定于已观测信号。

## 形成依据后的启用顺序

先由A确定目标/域、B给出变量及限值依据，D逐条写正常、边界、越界、缺失、错误单位和恢复用例。之后在独立仿真配置中验证，再形成新的版本与完整ProcessRules配置。该草案catalog不是ProcessRules/1.0，不能直接交给运行引擎。验证不足的条目继续关闭，已有R-01/R-02和执行保护继续生效。

检验草案一致性：

```bash
python deliverables/D01/scripts/validate_research_plan.py \
  --catalog configs/research/atropine_rule_candidates.draft.json \
  --protocol configs/research/atropine_collection_protocol.draft.json
```

通过仅代表草案字段与边界约定一致；输出仍为collection_authorized=false、R03_enabled=false、physical_thresholds_confirmed=false。
