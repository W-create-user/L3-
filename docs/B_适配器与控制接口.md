# B03/B04：适配器、变量字典与研究控制接口

## 选择后端

```python
from l3_system.execution import create_adapter, ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent

with create_adapter('MAbEnvGym', max_steps=10) as adapter:
    executor = ExecutionAgent(adapter, log_path='outputs/B_events.jsonl')
    executor.reset(seed=1610)
    monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
    # Generate a candidate; prepare_action never steps the environment.
    request = executor.prepare_action(
        {'task_id': 'example', 'task_type': 'control_cycle'},
        action=adapter._env.steady_actions)
    review = monitor.check_request(request, request['observation'])
    result = executor.execute_action(request, review)
```

`create_adapter('AtropineEnvGym', ...)`保留原Atropine实现。两个后端共同提供reset、get_observation、step、action_space、observation_space、state_version、step_count、step_calls、done、metadata、close。mAb依赖只在构造时导入，导入整个execution包不要求安装MPCTools。

执行请求的environment、normalize、uss_subtracted及backend_metadata来自适配器；不再把mAb标成Atropine。每次reset生成新episode_id并使旧state_version失效；请求准备和读取观察不推进模型。缺少metadata的第三方/测试替身明确标为unregistered_adapter，不猜测真实环境身份。

当前主CLI/多角色目标路由仍保持Atropine；mAb需显式使用B适配器与ExecutionAgent。A05生产目标路由、C训练接口和D工艺围栏不因本接口自动完成。示例里的D只提供已启用的接口/动作检查，不证明mAb工艺安全。

## 坐标和终止

| 后端 | 接口动作 | 物理转换 | 观察 |
| --- | --- | --- | --- |
| Atropine | 相对USS的偏置，normalize=false | physical=model+USS；进入plant后/1000转L/min | 原39维返回布局，E-factor不是减稳态的偏差 |
| mAb | physical/uscale，normalize=false | physical=model*uscale | physical/xscale；1970维，不能套用Atropine字典 |

`adapter.action_coordinates`提供to_physical/from_physical及to_normalized/from_normalized。后两者是单独的无epsilon线性[-1,1]传输坐标，**不是厂商normalize=true的epsilon变换**；实际后端始终normalize=false。不自动剪裁用户候选动作，D看见越界后可拒绝，适配器执行前再校验。

mAb厂商源码在计数自增前判断max_steps，存在多执行一步的边界。本适配器不改vendor，在成功调用计数达到显式预算时返回truncated=true、info.adapter_step_budget_reached=true，并阻止下一步。info.upstream_done保留上游done；采样周期按源码文档为60min。异常/错误奖励/无效观察不能计为成功；不确定结果后环境保持不可执行，不能自动重放。

## 字典与边界

[数据字典](data_dictionary.csv)逐项记录后端、索引、名称、模型坐标、比例/偏置、源码行号、模型Box及单位状态。包含Atropine 39观察+4动作、mAb 1970观察+9动作；来源和哈希在configs/environments/dictionary_provenance.json。

mAb前19项及9路动作的单位按源码注释登记；捕获段按CasADi 13×150列优先reshape映射为index=19+13*轴向节点+分量。q1/q2单位、最后积累量的物理质量转换未确认，不能补猜；不能把最后积分状态直接称为已捕获质量。所有模型Box均不是安全/可靠运行范围，所有变量的现场可测性和可靠工艺范围仍未确认。

Atropine流量单位由源码/1000进入L/min推断为mL/min，但上游绘图标签写uL/min，字典明确记录这一冲突；未声明现场标定。源声明单位也不等于整个化学模型单位一致性已审计。

configs/environments/mab_physical_registry.json是供A目标评价使用的**研究投影**：返回向量乘xscale后才可用dm评价buffer_height。它不是默认生产目标注册或可达/安全范围批准；未知单位的变量没有被加入目标变量集。

## 控制策略

SpaceMidpointPolicy保留接口基线；BufferLevelPID仅生成动作候选，不调用step。目标变量为mAb缓冲液位：观察17×xscale[17]为dm；动作7×uscale[7]为出口L/min。源码方程dh/dt=(F_in-F_out)/Ac；正液位误差应增大出口流量。

configs/controllers/mab_buffer_pi.json为冻结的研究PI候选（PID的Kd=0情形）。Kp来自上游_pcontroller的坐标换算；Ki是B明示的实验设定，不是经生产调参的参数。仅控制出口轴，其他轴采用源码稳态动作，不声称全流程或捕获最优。

有饱和限制和条件积分抗饱和；predict后必须commit(success)才能更新积分/微分历史。D拒绝应commit(False)，不把未执行候选当真实控制周期。setpoint、采样周期、调参初态及两个不同评价初态均在配置中声明；冻结参数后不使用评价初态调参。

研究目标为缓冲液位回到稳态15dm附近±0.05dm并保持3个真实step，由A的evaluate_goal接口计算，reward只留作原始记录。目标是B控制实验设定，未代A批准生产目标；safety_ok仅表示调用方通过已启用的接口检查，不表示工艺安全得到证明。MPC优化器尚未验证，不启用、不将失败求解动作放行。

## 安装与复现

使用Linux/Python3.12，在仓库根目录运行；以下outputs必须不存在以保留旧失败记录：

```bash
python3.12 -m venv .venv-atropine
.venv-atropine/bin/python -m pip install -r requirements-lock.txt
bash scripts/setup_mab.sh
.venv-mab/bin/python -m experiments.build_b_dictionary
.venv-mab/bin/python -m experiments.verify_b03_b04 --output outputs/b_live_new
.venv-mab/bin/python -m experiments.check_mab_guards --output outputs/b_guards_new
.venv-atropine/bin/python -m unittest discover -s l3_system/tests -t . -v
.venv-atropine/bin/python -m experiments.run_validation --output outputs/atropine_new
.venv-atropine/bin/python -m experiments.validate_session --output outputs/session_new
```

MPCTools固定来源及Python兼容补丁在patches；不要安装PyPI同名分析包。Windows原生仿真依赖/超时机制未验收；可在WSL使用Linux命令。

本分支按用户此前偏好只公开代码、模型字典及必要配置/接口文档，不新增上传原始轨迹或运行日志。运行脚本在用户指定输出目录保存完整证据。

## 与11月16日R-03批准的关系

configs/research/r03_experiment_authorization_20261116.json批准的是限定Atropine仿真实验，不含mAb，不是工业限值批准。verify_b03_b04只核验B负责的实时布局、坐标和阈值可观测性；不会启用R-03、替代A/C/D新鲜度与对账契约，或绕过运行启用门槛。
