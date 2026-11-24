# D01 数据接收与结构验收

本次把数据清单规范补成可运行入口。工具不依赖 GPU、企业授权或真实训练数据即可开发和测试；真正接收数据时仍要求 B 提供来源、字典、配置和独立标签。结构验收通过不表示物理单位、标签、权限和科研指标已经通过审核。

## 数据交付包

B 提供 manifest.spec.json、采集配置 collection.json、逐维字典 variables.json 和按 episode 保存的 JSONL。原始 B_events、扰动注入记录及标签文件另外保留，以便核对；不要用下面的模型输入规范覆盖原始证据。视觉、文本仍按既有清单引用、标记来源和计算哈希；本次深入检查的是数值时序内容，未自动验证图片内容或文本标签泄漏。

字典格式为 D01Variables/0.1，顶层包含 schema_version、environment、observation、action、review。两个向量列表逐项填写 index、name、unit、coordinate、source_reference，索引从0连续。review 包含 reviewed_by、reviewed_on、source_reference。不得用 UNKNOWN/PENDING 等占位值冒充单位；填写复核人只是声明，仍须真实人工审阅。当前只支持一维数值向量；更高维输入需先版本化扩展契约。

时序每行遵守 [step_record.schema.json](../step_record.schema.json)：episode_id、step_index、仿真时间及秒单位、带时区的采集时间、动作前后观测、候选及实际动作、executed、前后 state_version、reward、终止标记、执行状态及失败原因。确实知道仿真时间转换关系后才能填秒，不从环境名称或墙钟耗时猜测。每个 episode 可以分成多个时序文件，但清单中的文件顺序必须等于记录顺序。

step_index 是从0开始的交互记录序号，包含被拒绝的候选；此契约下 manifest.step_count 是该episode的交互记录数，不能当作 env.step 调用次数。实际调用数由 executed=true 的记录统计。executed 表示确实调用了仿真 step；调用失败或结果未知记 failed 并保留原始错误，不得改成 completed。failed 后本契约要求结束数据episode；恢复或reset后使用新episode并保留关联说明。

被拒绝的候选记录 executed=false，实际动作、动作后观测和reward均为null，state_version不变，不伪造终止。已执行且成功的记录必须有实际动作、后观测和reward；一次调用state_version递增1。这里的版本约定对应当前执行适配器，不应直接套到其他后端。raw info、标签和报警结论保存在独立原始证据中；标准输入行不允许额外标签字段。这项限制不能自动发现文本或图像中的隐含泄漏。

## 两步运行

从仓库根目录执行：

```bash
python deliverables/D01/scripts/prepare_manifest.py DATA/manifest.spec.json \
  --root DATA --config DATA/collection.json --dictionary DATA/variables.json \
  --output DATA/manifest.draft.json
```

生成器只按文件计算哈希，保留调用方提供的种子、初态哈希、场景、分组和标签来源，并强制写为 draft。它不批准来源或自动分配训练/测试集。输出不能覆盖已有文件。团队核对真实来源、授权和分集后，另存一份 status=ready_for_review 的清单，再运行：

```bash
python deliverables/D01/scripts/intake_dataset.py DATA/manifest.review.json \
  --root DATA --config DATA/collection.json --dictionary DATA/variables.json \
  --report DATA/intake_report.json
```

返回码0仅表示结构检查通过，返回码1表示失败。工具检查原清单契约、配置与字典哈希、字典完整性、时序维度、episode身份、记录数和顺序、状态版本连续性、终止后追加记录、前后时间顺序及拒绝动作是否伪造执行。正常/异常标签的真实性、种子和初态哈希的来源、配置与实运行是否一致仍须独立核对，不能靠手填字段证明。

这里的 DATA 是待提供的数据目录，不是仓库已经存在的数据集。示例命令不能在无交付文件时直接运行成功。examples/data_manifest.draft.json 仍是草案模板，不可改一个状态字段就宣称有可用训练集。

## 已有归档核对

可用 scripts/audit_execution_logs.py 盘点旧 B_events，检查请求/结果关联、重复结果和执行标记。该工具始终不把旧日志标为 training_ready。

本次核对 normal_1101、continuous_5 和 reject_1101 三份原有日志，来源固定在提交 090f5401fc886567a66b294110f9b2e7f6d91b9c。分别报告1、5、0次执行；每份都有1条reset记录。五步连续执行是一次reset后的五步，不是五条独立episode。结果存于仓库 evidence/d01_intake/archived_logs_audit.json；它是旧归档结构检查，不是新仿真或训练结果。

## 尚待确认

本次没有取得企业授权、GPU预约或 B 的字典签收，也没有替 A/B/C 确认数据契约。可运行工具与合成测试已具备，D01整体状态仍为待交接确认；D03训练继续依赖真正的正常轨迹及独立异常验证数据。
