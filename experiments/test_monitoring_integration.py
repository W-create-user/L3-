"""从仓库根目录运行：python experiments/test_monitoring_integration.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.monitoring_agent import MonitoringAgent, safe_step_request


class FakeEnv:
    def __init__(self):
        self.step_calls = 0

    def step(self, action):
        self.step_calls += 1
        return {'observation': [0.0], 'reward': 0.0}


def main():
    env = FakeEnv()
    monitor = MonitoringAgent()  # 演示模式，不是正式 AtropineEnv
    observation = {'temperature': 75.0}  # 仅占位；R-03 未启用

    ok = safe_step_request(env, monitor,
                           {'task_id': 'T001', 'action': [0.5, 0.2, 0.1, 0.0]}, observation)
    assert ok['safety_decision']['decision'] == 'pass'
    assert ok['executed'] and env.step_calls == 1

    bad = safe_step_request(env, monitor,
                            {'task_id': 'T002', 'action': [0.5, 1.5, 0.0, -2.0]}, observation)
    assert bad['safety_decision']['decision'] == 'reject'
    assert bad['safety_decision']['rule_id'] == 'R-02'
    assert not bad['executed'] and env.step_calls == 1

    missing = safe_step_request(env, monitor,
                                {'task_id': 'T003', 'action': None}, observation)
    assert missing['safety_decision']['rule_id'] == 'R-01'
    assert not missing['executed'] and env.step_calls == 1

    log_path = ROOT / 'experiments' / 'monitoring_test_logs.jsonl'
    monitor.export_logs(log_path)
    print('合法动作:', ok['safety_decision'])
    print('越界动作:', bad['safety_decision'])
    print('缺失动作:', missing['safety_decision'])
    print('FakeEnv.step 调用次数:', env.step_calls)
    print('日志已保存:', log_path)
    print('PASS: 三组测试全部通过')


if __name__ == '__main__':
    main()
