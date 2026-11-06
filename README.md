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

## 四智能体统一入口

已将 A 决策、C 调度、B 执行、D 监管接入 `runtime.cluster.AgentCluster`。
A 生成 HTN 子任务；C 按依赖通过 CNP 分配给具备能力的 A/B/D；B 准备候选动作；D 使用实际环境动作空间审核；只有审核通过后 B 才调用环境。每个子任务的完成或失败回传 C，保留父任务及子任务 ID。拒绝或执行失败时停止后续任务，返回失败原因及未执行的子任务状态。

运行接口测试（Python 3.10+）：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

运行仓库自带的真实 SMPL 环境：

```bash
python -m runtime.unpack_smpl
# 在独立虚拟环境中安装 SMPL 自身所需依赖
python main.py --goal 执行一次控制任务
python main.py --goal 检查状态
# 或使用已有的 SMPL 源码目录
python main.py --smpl-source /path/to/smpl
```

`requirements.txt` 只包含智能体接口测试所需依赖。真实环境还需安装 SMPL 的依赖（包括 requests、casadi、gym 等），可参考原 ZIP 中 `code/requirements-lock.txt`；锁文件含平台相关及历史版本，请根据实际平台创建独立环境。`runtime.unpack_smpl` 只提取原 ZIP 中的 SMPL 源码到 `vendor/smpl`。也可以设置 `SMPL_SOURCE_DIR`。

自定义动作和日志位置：

```python
from runtime.cluster import AgentCluster
from runtime.environment_adapter import EnvironmentAdapter

with EnvironmentAdapter() as env:
    cluster = AgentCluster(env, log_path='work/events.jsonl')
    result = cluster.run('执行一次控制任务', seed=1101)
    # cluster.run('执行一次控制任务', action=[...]) 传入符合真实动作维度的向量
    print(result['success'], result['subtask_statuses'])
```

每次 `run` 重置环境，代表独立单次任务。默认动作策略沿用 B 的动作空间中点基线，不代表工艺优化控制器。D 当前启用输入合法性和动作边界规则，尚未配置工艺状态阈值。

源码来源：`agents/decision_agent.py`、`agents/scheduling_agent.py` 和 `algorithms/cnp_protocol.py` 从仓库对应 A/C 目录整合；`runtime` 下执行、环境适配、任务契约、安全桥和策略模块来自原 B 压缩包；D 沿用根目录监管实现。原始交付目录和 ZIP 保留作为来源记录，统一入口使用上述整合后的模块。

验证记录：6 项确定性接口测试通过，使用测试环境适配器，覆盖四智能体闭环、拒绝时零步进、只读零步进、失败停止、重复任务及过期请求。当前云环境缺少 `requests`，真实 SMPL 启动未通过，因此接口测试结果不等同于真实工艺仿真验证。

## 中期智能体系统补充（SMPL路线）

详见 `docs/智能体系统中期报告.md`。新增 `runtime.dispatch.Dispatcher` 的多候选投标和有界重分配、`runtime.supervision.Supervision` 接管锁存、`AgentCluster.run_batch`、带来源的状态规则配置及 `runtime.gym_adapter.GymEnvironmentAdapter`。目前17项测试通过。

```bash
python -m unittest discover -s tests -v
python -m experiments.evaluate_system
```

实验输出为 `experiments/results/system/{raw.csv,events.jsonl,summary.json}`，全部是合成接口验证，不能作为真实SMPL工艺性能数据。项目按新版项目书采用SMPL：mAbEnv主研究、AtropineEnv辅助验证；真实环境验收状态和剩余缺口以本次中期系统报告为准。
