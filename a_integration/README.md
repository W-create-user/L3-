# 最新 A 交付包的仓库集成版本

来源：用户上传 `L3_A_Final_20261108.zip`，包内顶层目录为 `L3_A_Integrated_20261106`。文件名中的日期作为交付标识，不代表本次验证日期。原包校验与集成调整见 `source_manifest.json`。

## 运行

在仓库根目录，使用已安装原 SMPL 运行依赖的 Python：

```bash
python main.py --engine a-final --offline-preview
python main.py --engine a-final --seed 1101
python -m a_integration.main --seed 1101 --output-dir outputs/a-final/new-run
python -m unittest discover -s a_integration/tests -v
```

输出目录必须不存在，避免覆盖证据。默认生成带随机后缀的新目录。原入口 `python main.py --scenario normal`、`reject`、`inspect` 继续使用已有集成；新 A 入口通过 `--engine a-final` 选择。

`a_integration` 使用包内新版 DecisionAgent、SchedulingAgent、D v1.2 及接口适配和重规划模块。B 执行模块和 SMPL 模型统一复用根目录 `执行智能体/code`，不再复制 vendor、B代码或字体缓存。各模块导入使用命名空间，避免覆盖已有 `agents` 或 `main`。

## 解析与集成内容

- DecisionAgent 使用统一 task_type、小写任务状态，增加前置条件和完成判据元数据。
- A→C 保留父子任务ID、依赖与约束，检查分配获奖者身份。
- D v1.2 提供 check_request、审核日志导出，与 B 最终安全门控衔接。新版控制路径先做预审，B执行前再审核；不是将旧入口的单次审核证据套用于新版。
- ReplanningController 提供有界重分配策略和安全拒绝分类；真实入口不会在缺少兼容备用执行者时自动重放工艺动作。
- 支持控制六步与状态检查三步。原包仅有控制处理器，本次补齐状态检查分支。

## 验证与边界

本次35项测试通过，其中34项来自代码包，1项新增只读状态检查回归；测试使用真实 B 执行类、新版 D 和 FakeAdapter，属于接口合同测试。另通过Python编译检查和根目录离线预览。输出见 `evidence/a_final_integration`。没有在本次集成中重新运行真实SMPL，原 `evidence/four_agents` 保留为旧入口的历史实测证据。

投标为配置档案，不是实时独立执行者报价；中点策略不是优化控制器。前置条件/完成判据目前是有限模板合同，不能解释为通用条件求解器。R-03、mAbEnv实跑和工业L3达标未完成。非执行阶段异常收口、超时取消和持久化恢复仍需完善。

`docs/` 内为原包技术说明，路径和原始测试数字描述原包布局；在本仓库运行以此README为准。原包B/C/D历史证据未重复导入，避免被误计为本次新增实测。
