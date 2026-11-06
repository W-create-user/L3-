成员 D → A/B/C 联调接口（v1.2）
文件
- agents/monitoring_agent.py：MonitoringAgent、safe_step、safe_step_request
- agents/safety_rules.py：R-01、R-02 规则函数，R-03 尚未启用
- experiments/test_monitoring_integration.py：模拟环境联调测试
A 的调用方式
from agents.monitoring_agent import MonitoringAgent, safe_step_request
monitor = MonitoringAgent(action_space=env.action_space)  # 真实 Box 环境
request = {'task_id': 'T001', 'action': candidate_action}
result = safe_step_request(env, monitor, request, observation)
decision = result['safety_decision']
if decision['decision'] == 'reject':
    # 将 decision 原样反馈给 A 的任务失败/重规划流程
    print(decision['task_id'], decision['rule_id'], decision['reason'])
- 仅查询：`monitor.check_request(action_request, observation)`；此方法不调用 env.step()。
- 受控执行：`safe_step_request(env, monitor, action_request, observation)`；仅 pass 时调用 env.step()。主程序不得在拒绝后绕过门控直接调用 env.step()。
- ActionRequest：`{'task_id': 非空字符串, 'action': 数值数组/列表}`。
- SafetyDecision：`{'task_id': str或None, 'decision': 'pass'/'reject', 'rule_id': str或None, 'reason': str, 'risk_level': int}`。
- observation 同时传入；当前 R-03 未启用，不会根据状态判断。
- R-01：缺失/格式错误/维度不符/NaN/Inf → reject，Level 3。
- R-02：动作超出 action_space.low/high → reject，Level 3。
- 来源：R-01 是项目输入完整性规则；R-02 边界来自环境 action_space，不等于经验证的工业安全阈值。
- 演示参数 4 维、[-1,1]，仅供测试；真实 AtropineEnv 的动作空间与状态字段仍待 B 提供。
- 当前只实现 Level 0 放行与 Level 3 拒绝；Level 1/2、R-03、Web 预警、紧急停机未实现。
- 运行 python experiments/test_monitoring_integration.py 生成 experiments/monitoring_test_logs.jsonl；该日志属于 FakeEnv 模拟测试。
说明
本交付包仅更新 D 负责的文件。请不要覆盖 A/B/C 的现有代码。提交共享仓库前先检查 git status，并遵守团队分支/PR 约定。