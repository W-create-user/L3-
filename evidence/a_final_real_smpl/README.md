# 新版A入口真实SMPL验证

本轮在Linux、Python 3.12.14中运行仓库提交22af2c3的新版A控制入口，未修改SMPL模型方程。Atropine模型及配置文件的Git blob哈希与仓库一致。

| 场景 | 种子 | 六步/三步结果 | 真实step增量 |
|---|---|---|---|
| 正常控制 |1101|六步全部完成|1|
| 正常控制 |1102|六步全部完成|1|
| 正常控制 |1103|六步全部完成|1|
| 明确越界候选 |1101|R-02拒绝，后续不执行|0|
| 状态检查 |1101|三步完成|0|

总计3次真实step。39维观测、4维动作，D demo_mode=False。中点策略用于接口验证，不代表优化控制、mAbEnv、R-03或工业L3达标。

normal_1101 为根目录 CLI 实跑，其余由 experiments/run_a_final_validation.py 生成。每个目录保留 A_summary.json、A/B事件和D审核日志。validation.json 记录后四个场景、代码哈希及环境；首个场景结果见 normal_1101/A_summary.json，不能重复统计。

实际NumPy模块为1.26.4；依赖目标目录曾保留2.5.4的旧distribution元数据，因此manifest同时记录该差异，NumPy版本以模块导入值为准。CasADi使用已装版本3.7.2，并非Windows历史环境的3.8.2。日志保留Gym、AMP和CasADi警告。

依赖下载与本地模型HTTP服务需要正常网络和回环套接字权限。本轮只补齐执行依赖，使用CPU PyTorch。前两次准备阶段因缺依赖或本地套接字权限失败，均未形成成功仿真结论；首次成功运行是normal_1101。

复现（已安装SMPL依赖的Python）：

```bash
python main.py --engine a-final --seed 1101 --output-dir outputs/new_cli_run
python experiments/run_a_final_validation.py --output outputs/new_validation_run
```

所有输出目录须为新目录；不会覆盖本轮证据。
