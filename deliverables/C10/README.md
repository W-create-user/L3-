# 成员 C 11 月 27 日联调交付

本交付提供独立的 C 入口，调用仓库现有 A 决策、B 执行与 D 审核。A 驱动会话，分别通过本机 ZeroMQ 与 C 调度、B 执行、D 审核进程通信。只有 B 进程持有环境，现有动作请求与双次 D 审核仍生效。C 在传输边界为授予附加编号、尝试号和 10 秒租约。反馈必须关联当前授予，过期租约保留锁并要求对账。主链的默认本地启动方式由 A 维护。

| 验收 | 实测 |
| --- | --- |
| 正常 Atropine 四进程 | A 的六级任务链完成，B 的真实 SMPL 仿真步进 1 次。相同执行请求返回同一回执，冲突请求拒绝，计数仍为 1。 |
| C 进程退出 | 测试在首轮锁释放后停止 C，下一轮注册超时，A 进入接管，继续发起周期被拦截，B 计数仍为 1。 |
| 恢复和拒绝 | C 重启应答、模拟操作员确认后，下一轮使 B 计数增至 2。随后越界候选由 D 拒绝，计数保持为 2。 |
| 学习策略门控 | 复载 C07 选定权重并校验 SHA-256；两 actor 各读训练坐标中的一个本地标量。通过同一 A 流程、B 执行和 D 审核跑合成控制环境，成功步进 2 次；越界候选由 D 拒绝、零额外步进。该权重保留为研究样本。 |
| B mAb 联合动作接口 | 在 B 的固定只读版本运行真实 mAb。稳态参考动作经组件收集及 D 动作审核后执行一次，越界分量审核拒绝、零额外步进。研究 wrapper 的默认训练入口明确拒绝，允许的探针也返回 `learning_eligible=false`。 |
| 回归 | 全新虚拟环境安装项目依赖后，181 项主链、9 项 C 专项通过；在各自只读检出目录，13 项 B 网关和 6 项 D 开发门控测试通过。 |

C 进程退出复验发生在已对账的周期边界。B 在运行中丢失回执仍属于未确认执行状态，入口不会自动重试。进程内执行回执只在本进程存活期间可复用。逻辑多资源调度使用 C08 入口并保留配对证据。正式 A08-MAB-03 仍按团队指定的本地单执行资源候选管理，分布式 Atropine 演示与 mAb 网关探针是各自独立的开发证据。

## C 侧核验记录

C 已验证公开夹具上的资源回收、授予关联、过期锁保留、B 唯一步进和故障恢复。C 主张保持 A08-MAB-03 首轮的单资源与现有范围。A 的冻结文件尚无正式运行 SHA、实际生效的审核配置哈希和联合签署。B 候选尚未开放 mAb 学习转移，D 的 D26 gate 发布范围仍为开发验证。正式 mAb MADDPG 门控和 A/B/C/D 对同一正式版本的核签仍待成立。本交付未访问正式保留组。

## 复现

从 C 分支的仓库根目录，在安装 libzmq 4.3.5 的 Linux 或 WSL 中运行。

```bash
python -m venv .venv-c
source .venv-c/bin/activate
python -m pip install -r evidence/c_20261127_integration/python-dependencies.txt
python -m unittest discover -s l3_system/tests -t . -v
python -m unittest discover -s deliverables/C08/tests -v
python -m unittest discover -s deliverables/C10/tests -v
python -m deliverables.C10.live --out /tmp/l3-c-atropine
python -m deliverables.C10.live --backend learning-fixture --out /tmp/l3-c-learning
(cd evidence/c_20261127_integration && sha256sum -c SHA256SUMS)
```

mAb 探针在 B 的 `68831ef9e06440a1066e9d18bfb3955e1ac3006f` 检出目录运行。MPCTools 取 `ac683dc8199d3e11bf4f3971a3b8f613c83a162b`，应用 B 历史提交 `dd76ceafb1f1b331660d3a171cbcd087fd0b7559` 的 `patches/mpctools-python312.patch`，再把 C 的仓库根目录和 MPCTools 目录加入绝对路径形式的 `PYTHONPATH`。

```bash
python -m deliverables.C10.gateway_probe --out /tmp/l3-c-mab-gateway
```

[调度和学习方法章节](../C08/README.md) · [本次联调原始日志、依赖版本及校验值](../../evidence/c_20261127_integration/)

## 11 月 27 日补充故障验证

`python -m deliverables.C10.live --lost-reply --out /tmp/l3-c-lost-reply` 在真实 Atropine 上延迟 B 的首次执行回包至 A 的接收截止后。B 已步进一次。A 进入接管，C 保留锁，提前恢复与过期请求号对应的回执均被拦截。随后从仍存活的 B 进程只读查询原请求结果，核对版本与计数，确认原动作已执行。C 完成对账并释放锁，模拟操作员确认后才派发新一轮。复验没有重放原动作。B 进程自身丢失或账本丢失仍需人工处置。

C08 的动态投标带独立的签发和失效时间，调度端拒绝过期、尚未生效及非有限的时间字段；既有同步固定投标保持兼容。重跑 90 次调度对照后，统计 CSV 与首轮一致。补充复验中 181 项主链、10 项 C 专项通过。[原始日志和校验值](../../evidence/c_20261127_followup/)
