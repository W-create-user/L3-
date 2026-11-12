# B → D 三项交付（2026-11-12）

本目录按用户最新要求公开上传 GitHub。数据为科研仿真，不是现场数据。记录来自11月12日本次真实运行，不伪装成11日运行。
源代码运行基线：`2e88c200afa4698b0dc13a9d2260fbaf44d56972`；新增探针工具：`d13c69b986564019a253e507d4d880c3025d4626`。本分支后续提交只是整理交付，不是重新仿真的版本。

## 1. 原始schema、动作、单位与尺度

- [observation_schema.json](observation_schema.json)：用户上传中期ZIP中 `data/principal_300/observation_schema.json` 的原始字节副本，39个Atropine观测字段，保留原有“不含安全阈值”标识。
- [original_exports/observation_schema_20261104.json](original_exports/observation_schema_20261104.json)：ZIP内11月4日导出版本原文件，不与principal_300版本混同。
- [observation_schema_verified.json](observation_schema_verified.json)：当前源码/输出维度和排列核验，另加单位可信度、现场传感器对应关系是否确认。不改写原schema。
- [action_dictionary.json](action_dictionary.json) 与CSV同名表：Atropine 4路及mAb 9路动作；名称、上下界、基准偏置和缩放公式，保留未确认项。
- [mab_space_metadata.json](mab_space_metadata.json)：mAb实测1970维观测/9维动作；尚未有逐变量字典，不能套用39维Atropine schema。

Atropine：本次normalize=false、uss_subtracted=true，执行输入为候选增量+USS，进入plant后再/1000；mL/min是由源码L/min和/1000推导，未现场标定。观测的第0项是自增前的step_count，第一次step后仍可能为0，不能当分钟。
mAb：normalize=false仍是physical/uscale，物理动作=model_action*uscale；前8路单位尚待B03逐变量确认。最后一路为捕获柱切换量，源码>=0.5触发切换。动作边界不是现场安全认证。

## 2. 最新配置、原始轨迹、种子、扰动和标签

- [run_configs.json](run_configs.json)：6个Atropine运行的种子、初始化观察、策略、开关、采样周期、步数和复现命令；3个正常单步seed=1101/1102/1103，连续5步seed=1101，拒绝/只读seed=1101。
- `runs/<run_id>/B_events.jsonl`：未经改写的原始reset、ActionRequest、SafetyDecision和ExecutionResult。`events.jsonl`、`D_reviews.jsonl`、`result.json`为配套原始记录。
- [trajectories.jsonl](trajectories.jsonl)：从原始B事件派生的方便读取视图，含请求动作、实际执行动作（被拒绝时为null）、前后观察、step计数、源文件和行号。原始B事件是权威记录。
- [disturbances.json](disturbances.json)：每个运行的初始化随机化与扰动情况。没有注入过程扰动或传感器噪声；reject场景仅把候选动作设为high+1，D拒绝后没有推进模型。不能将拒绝轨迹当成执行越界后的工艺响应。
- [label_provenance.json](label_provenance.json)：标签来自具体运行行号的D判定/B结果。`interface_baseline_pass`仅表示接口执行成功，`action_bounds_rejection`仅表示动作越界请求被拒绝；`process_anomaly_label=null`表示缺少工艺异常真值。只读无执行标签。
- [runs/atropine_baseline/report.json](runs/atropine_baseline/report.json)：另一seed1112、max_steps=2的原始reset与两步记录，第2步按步数上限终止。
- [runs/mab_probe/report.json](runs/mab_probe/report.json)：seed1112、初态偏差0、max_steps=2，真实reset和1次step；逐阶段耗时、内存、原始观察与动作均在文件中。没有注入扰动，没有工艺异常标签，不证明连续10步能力。
- 两个requirements锁文件用于说明运行依赖；MPCTools须按仓库scripts/setup_mab.sh固定来源并应用patches中的补丁。

主Atropine派生视图只有8次成功step及1次拒绝，不足以训练R-03。即使seed不同，确定性/近确定性初始化也不能冒充独立随机样本。没有三类工艺异常数据集，缺口明确保留。
仿真时间按实际成功步数乘10min派生，不是JSONL墙钟时间，不直接读取观察第0项；被拒绝时仿真时间不增加。

## 3. 已确认规则与未确认阈值

[rule_thresholds.json](rule_thresholds.json)列出源码提交、文件、行号与状态：

| 规则 | 依据/状态 |
| --- | --- |
| R-01 | 已确认软件输入契约：缺失/形状错误/非数值/NaN/Inf拒绝 |
| R-02 | 已确认Atropine live Box边界及严格越界比较；是模型允许输入范围，不是工业安全阈值 |
| STATE-INVALID | 已确认观察维度与有限值完整性检查，不是工艺异常报警 |
| R-03 | **未确认、未启用、阈值null**。需要D与工艺依据共同制定 |
| mAb切换0.5 | 已确认模型动作逻辑；不是工艺危险阈值，也未接入现有D监管 |
| 质量/能耗/现场安全限值 | **未确认**；没有可引用的现场标定/审批依据 |

原schema的Box、经验极值、reward误差和步数终止都不能自动变成报警阈值。Atropine阈值不得套用mAb。

## 校验与复现

在仓库根目录运行 `python scripts/validate_D_handoff.py`，检查所有文件SHA256、原schema原字节身份、维度、请求/执行动作关联、拒绝不推进以及未确认阈值状态。
`manifest.json`保存交付文件逐个哈希；`provenance.json`说明来源与缺口。
本次未代D确认验收。D可在本目录直接读取所需文件，无需先合并main；三个交付项已提供，未确认的工艺结论明确保持未确认。
