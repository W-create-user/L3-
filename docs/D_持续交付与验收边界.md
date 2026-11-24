# D 持续交付与验收边界

本次承接11月23日首次交付，继续完成 D 可独立实施的监管、实验与复现工作。A、B、C 的受检源码保持原样；mAb 在独立组合副本运行，原 B 工作树没有修改。历史提交、失败实验和中间模型均保留。

## 已完成的工程交付

| 工作 | 实际证据 |
| --- | --- |
| 检测误报改进 | 采用孤立森林分数与参考相对残差组合；训练、概率校准、阈值选择按源episode分别划分。最终模型仅用训练和验证拟合。 |
| 新保留测试 | 三轮各12条、720步，共新增36条、2160步。前两版未达标并保留；最终版使用新的48001—48012初态，锁定后评价一次。 |
| 置信度工程 | Isotonic校准只使用验证中的独立源组，概率含义限定为人工传感器偏置窗口标签。模型不输出物理安全概率。 |
| 三级风险与Web接口 | D独立入口组合现有A/B/C代码，接通受限采样、人工确认、自主执行、硬规则拒绝和明确新episode恢复。29个HTTP检查通过，实际16次Atropine step。 |
| 三模态训练 | 补用官方torchvision ResNet18 IMAGENET1K_V1权重；3个初始化种子、7种对照，共21个模型各20轮。所有模型保存恢复的验证输出逐值一致。 |
| mAb模型围栏 | 读取B的mAb适配器和模型定义，验证buffer索引17、xscale=3、物理单位dm；6个边界/单位/坐标检查和3次真实step通过。 |
| D实验协议 | 保存本轮传感器实验的锁定模型、条件、种子、预算、判据和指纹；完整项目的三系统生产对照仍保持未冻结。 |
| D统计与方法材料 | 完成配对episode误报分析、条件区间和监管方法章节，保留样本量与独立性限制。 |
| 清洁环境复现 | 新Python环境503项测试通过；最终检测模型重新训练后字节哈希一致。没有冒称独立人员签字。 |

## 最终检测结果及限制

最终保留测试共684个相关滑窗：TP=96、FP=0、FN=0、TN=588；原始注入事件检出6/6；整条无注入轨迹误报0/6。Brier=0.00017523，10箱ECE=0.00131459，均为本小样本相关窗口的诊断量。

原始高误报基线没有覆盖或删除。第一版新检测器在28001—28012测试中的窗口召回为78.125%，未达到预先登记的80%工程门槛。第二版在38001—38012上仍未达标。检查发现阈值候选只包含观测分数，不能表达“最大负例分数之后”的开区间边界。最终实现补齐边界候选，按验证集零误报约束与最低可行阈值选择；随后另采48001—48012测试，未复用前两轮作为未查看测试。

最终结果仅支持本次受检Atropine、中点控制、人工乘性偏置范围的工程检查。没有证明广泛工况、现场传感器、物理异常或生产目标泛化。即便暂按独立episode假设，6/6事件检出的95%二项区间约为[0.541,1]，0/6整段正常误报的区间约为[0,0.459]。独立代表性假设尚未成立，不能据此认证总体可靠度。

三模态数据仍是同一数值轨迹的因果派生图像和文本，官方预训练不改变这一来源。20轮为固定计算预算，不能当作收敛证明。七组结果没有支持“融合全面优于单模态”；测试每个模型86个抽样窗口，结果详见机器报告。没有根据测试择优选种子或继续追加轮数。

## 风险干预与恢复语义

启动保持受限状态；每次采样须提交操作人及原因，并通过D硬规则复核。系统不偷偷执行warmup。4条有效post-step观测组成第一个窗口，连续3个有效模型判定后才允许人工确认；确认本身不调用step。后续候选仍走D审核和B执行协议。

硬规则拒绝、上下文过期、模型不适用、未知执行结果和工况配置变化均阻断推进。严重风险后，同一条冻结观测不能反复计数为恢复样本。本仿真入口要求明确开始新episode并重新积累历史，日志中与旧episode分开；这不是同一生产过程的物理修复。受限采样、模型域与新episode策略都是隔离仿真设定，不是工厂停机策略。

入口：`python -m l3_system.monitoring.supervision_console --model 模型文件 --expected-model-sha256 已核验哈希 --output 新输出目录`，默认仅监听127.0.0.1:5056。该入口属于D模块，原A界面和主入口未改。

## 仍不能签为完成的项目

| 项目 | 具体依据或阻塞 |
| --- | --- |
| 浏览器展示及实录 | 本地Chromium启动被环境拒绝；云浏览器无法访问本地服务，随后访问策略拒绝。没有绕过限制，没有生成或冒充浏览器录像。已有HTTP和真实模型日志，以及可运行的浏览器验收脚本。 |
| mAb物理安全R03 | 已核验模型范围与单位，但没有现场危害阈值、专家签收或物理异常真值。模型边界不能改名为工厂安全极限。 |
| 独立视觉和现场文本能力 | 当前是派生模态，未取得可核验独立现场数据。 |
| 完整项目D07及生产实验 | 仍缺三组生产控制器、三类生产工况与目标/资源定义的联合冻结。不能把本轮传感器基准替代生产成功率90%或效率提升20%的证明。 |
| 论文发表 软著授权与最终独立签字 | 没有相应真实回执、授权文件或独立复核人；不能由代码或本次自测补写。 |

## 复现与证据入口

先按仓库原要求安装核心依赖、`requirements-monitoring.txt`、D01依赖及CPU torch/torchvision。运行测试：

```powershell
.\.venv\Scripts\python -m unittest discover -s l3_system/tests -t . -v
.\.venv\Scripts\python -m unittest discover -s deliverables/D01/tests -v
.\.venv\Scripts\python -m unittest discover -s deliverables/D04/tests -v
.\.venv\Scripts\python -m unittest discover -s deliverables/D06/tests -v
.\.venv\Scripts\python -m unittest discover -s deliverables/D07/tests -v
```

原1920步数据在首次实验记录包；本次包包含新增保留轨迹、三个模型版本、官方权重来源记录、三模态共享主干和训练头、HTTP/真实mAb记录及清洁环境复现。解压后以实际目录替换下列路径：

```powershell
.\.venv\Scripts\python -m l3_system.monitoring.residual_train --dataset 原1920步数据目录 --binding-record 本次包/residual_v3/freeze.json --output outputs/d_residual_reproduced
.\.venv\Scripts\python -m deliverables.D06.scripts.verify_supervised_http --frozen 本次包/residual_v3 --output outputs/d_http_reproduced
.\.venv\Scripts\python -m deliverables.D08.scripts.analyze_monitoring --source 本次包/holdout_v3/summary.json --output outputs/d_analysis.json
```

所有输出目录/文件须为新路径。模型pickle只加载本次可信训练产物，且检查外部已核验哈希。原始轨迹和模型保留为私有交付，公共仓库只含汇总、配置、源码、兼容补丁与证据指纹。

机器证据见 `evidence/d_completion/`；源分集与初态复核见 `lineage_review.json`，68个源episode具有68个不同初态指纹，不据此声称统计独立。D监管实验冻结记录为 `configs/research/d_monitoring_frozen_20261123.json`，完整项目草案状态没有被改成已冻结。
