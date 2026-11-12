# L3 成员D首批实施交付

2026年11月12日，依据《01_L3_总执行计划与任务分析(1).docx》与《05_L3_D_详细任务报告(1).docx》执行。今天推进 D01，并准备 D02 的输入与验收用例；后续 D03 至 D10 保留总计划的依赖顺序。

本包已落地规则来源表、数据清单 Schema、分集与三模态来源规范、可运行清单校验器、反例测试和资源缺口登记。D01 的规范与工具已形成首版，企业数据授权、GPU时段、B03变量字典及B05数据仍待核实，因此 D01 尚不能整体标为验收完成。R-03、孤立森林训练、多模态融合本次均未完成。

## 与真实仓库的对应

本次通过 GitHub 连接读取 main，仍为提交 [2e88c200afa4698b0dc13a9d2260fbaf44d56972](https://github.com/W-create-user/L3-/commit/2e88c200afa4698b0dc13a9d2260fbaf44d56972)。本目录是 D01 的独立工具与文档，不复制既有系统源码。evidence/source_index.json 记录本次核验的原仓库路径、Git blob SHA 和固定版本链接；evidence/d01_validation.json 中的源码哈希检查来自此前本地快照核验。

| 检查项 | 本次确认的事实 | 对D任务的影响 |
|---|---|---|
| R-01 | validate_action 检查可转换性、shape与有限值；check_request 检查 task_id | 已有接口规则；并非工况异常识别 |
| R-02 | 按传入 action_space 的 low/high 检查，等号允许 | 保留动态读取，不硬编码演示边界 |
| STATE-INVALID | 检查数值类型、有限值；配置 observation_space 时校验shape | 数据完整性，不等于物理安全 |
| R-03 | RULES 与 check_action 明确未启用 | 等待 B03/A02，不自行填阈值 |
| 会话恢复 | runtime.resume 已有人工字段、锁、环境就绪及观测检查 | D06扩展验证已有接口 |
| 离线样本 | safety_samples.csv 为80行、40候选、20越界候选 | 非80条episode，不可充作D03训练集 |
| 历史测试 | session_integration/summary.json 记86通过0失败 | 是归档值，本次未重跑这些测试 |

需要保留的实现细节：validate_action 使用 np.asarray(dtype=float)，因此可转换的数字字符串和布尔值并不会在该函数中被统一当作类型错误拒绝。不得把 R-01 描述成“拒绝所有非原生数值类型”。执行适配器另有 dtype 检查，两者不能混为同一条判断。

check_action 仅在构造器提供 observation_space 时执行观测检查；main 的示例入口提供了该参数。单独使用 MonitoringAgent(action_space) 不能声称拥有同样的观测shape拦截。check_observation 则可独立调用。

Atropine 源码有来自数据集或约定的观测范围；这些数值不能自动作为独立工艺安全依据。适配器 normalize=False，上游动作还涉及 uss_subtracted，物理单位及绝对值/偏差映射等待 B03 核验。

原始《项目修改后的版本.doc》已转换并核对第六节和第八节：第六节要求三模态算法，第八节声称企业数据和GPU基础。本次未收到对应授权书、实际数据、GPU设备或预约证明，全部登记为未核实，不推断团队线下没有这些资源。

## 使用方法

从仓库根目录先执行 `cd deliverables/D01`，再在本目录运行，需 Python 3.10+ 及 requirements-d01.txt 中依赖：

```bash
python -m unittest discover -s tests -v
python scripts/validate_manifest.py examples/data_manifest.draft.json --root examples
```

第二条应返回退出码1：草案含待填哈希且无真实episode，不能通过验收。B05交付后，按照 data_manifest.schema.json 建立实际清单，将 --root 指向模态文件共同根目录，再执行校验。真实清单 status 应为 ready_for_review，通过校验只代表满足部分技术契约，还须人工审查数据权限、变量单位与无标签泄漏。

本次新增测试只验证清单契约与泄漏防护逻辑，使用临时合成文件，既没有运行 Atropine/mAb，也没有进行模型训练。运行记录在 evidence/d01_validation.json；不能将这些测试计入仓库已有86项并声称系统测试达到108项。

## 首批交付与后续顺序

| 任务 | 已产出 | 状态与下一步 |
|---|---|---|
| D01来源登记 | docs/safety_rule_sources.csv 与资源核验表 | 源码规则已核对；外部依据等待确认 |
| D01分集规格 | docs/data_split_protocol.md 与Schema | 首版拟定，待B/C确认采集契约 |
| D01数据验收工具 | scripts/validate_manifest.py 与测试 | 已本地验证；尚无实际训练数据通过验收 |
| D02准备 | docs/D02_dependency_checklist.md | 待B03/A02，尚未启用工艺规则 |
| D03至D04 | 已保留数据与标签边界 | B05交付后训练；不以现有离线候选数据代替 |

下一步先接收 B03 的变量字典和依据，逐条实现 D02；17日 B05 首批真实仿真轨迹交付后进行 D03 正常训练，再推进 D04 三模态。D05风险融合与D06接管验收依赖这些输出，不能跳过。D07在24日前冻结正式实验口径，D08使用实际轨迹统计，D09/D10按原始证据审核。

本目录为 D01 增量，提交到独立分支供 PR 审阅。现有 l3_system 运行逻辑未修改；校验工具尚未接入主运行链。

## D02 后续进展

2026年11月12日继续实现了可配置过程围栏并接入既有审核链。详见 [D02说明](../../docs/D02_工艺规则与运行围栏.md) 和 evidence/d02/summary.json。工程规则已有合成测试，默认 Atropine 配置仍关闭；物理阈值、B03/A02输入、真实环境验收与D01外部资源核验依然待完成。上文首批交付状态描述保留为历史记录。
