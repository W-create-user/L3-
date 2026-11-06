## 四类智能体统一入口

新增根目录 `main.py`，接入实际 A/C/B/D 与真实 AtropineEnv，提供正常、拒绝、只读三种运行。请以 [统一运行说明](docs/四智能体运行说明.md)、[更新后的中期报告](docs/项目中期报告.md) 和 [最终实验记录](evidence/four_agents/summary.json) 为准。旧交付快照中的“尚未集成”等表述保留作历史记录，不代表本次最新状态。当前中点策略不是优化控制，mAbEnv、多模态与工业L3未验证。

# L3-# 面向流程工业 L3 级自主运行的智能体集群多模态监管与协同执行研究

项目编号：S202610703  
申报单位：西安建筑科技大学

## 项目简介

本项目面向流程工业 L3 级自主运行场景，研究智能体集群的多模态状态监管、任务协同和安全执行方法。

## 主要模块

- 智能体集群
- 多模态信息处理
- 运行状态监管
- 任务规划与协同执行
- 实验评估

## 当前进度

项目仓库初始化，正在进行需求分析和技术调研。

## 仓库结构

- `docs/`：项目文档
- `src/`：源代码
- `tests/`：测试代码
- `experiments/`：实验记录
- `data/`：数据说明

## 最新 A 代码包集成

已接入用户交付的 `L3_A_Final_20261108.zip`，位于 [`a_integration`](a_integration/README.md)。

```bash
python main.py --engine a-final --offline-preview
python main.py --engine a-final --seed 1101
python -m unittest discover -s a_integration/tests -v
```

原四角色入口保持默认路径。新版A复用仓库B和SMPL代码；本次35项接口测试通过，不将测试适配器结果写成新增真实工艺实验。
