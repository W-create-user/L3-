"""成员 D：MonitoringAgent v1.2，面向成员 A 的 ActionRequest/SafetyDecision 联调。"""
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

try:
    from .safety_rules import validate_action, check_bounds, observation_integrity_error
    from .process_rules import ProcessRuleEngine
except ImportError:  # 兼容直接运行 python agents/monitoring_agent.py
    from safety_rules import validate_action, check_bounds, observation_integrity_error
    from process_rules import ProcessRuleEngine


class MonitoringAgent:
    def __init__(self, action_space=None, observation_space=None, *,
                 process_rules=None, process_context_provider=None):
        self.logs = []
        self.observation_shape = (tuple(observation_space.shape)
                                  if observation_space is not None else None)
        self.demo_mode = action_space is None
        if self.demo_mode:
            self.action_shape = (4,)
            self.action_min = np.full(4, -1.0)
            self.action_max = np.full(4, 1.0)
            print('[WARNING] 演示模式：4维 [-1,1] 不是 AtropineEnv 真实边界')
        else:
            if not all(hasattr(action_space, x) for x in ('shape', 'low', 'high')):
                raise TypeError('仅支持具有 shape/low/high 的 Box 动作空间')
            self.action_shape = tuple(action_space.shape)
            self.action_min = np.asarray(action_space.low, dtype=float)
            self.action_max = np.asarray(action_space.high, dtype=float)
        if (self.action_min.shape != self.action_shape or
                self.action_max.shape != self.action_shape or
                np.any(np.isnan(self.action_min)) or
                np.any(np.isnan(self.action_max)) or
                np.any(self.action_min > self.action_max)):
            raise ValueError('动作空间上下界配置不合法')
        self.action_dim = int(np.prod(self.action_shape))
        if process_rules is not None and not isinstance(process_rules, ProcessRuleEngine):
            raise TypeError('process_rules must be a ProcessRuleEngine')
        if process_context_provider is not None and not callable(process_context_provider):
            raise TypeError('process_context_provider must be callable')
        self.process_rules = process_rules
        self.process_context_provider = process_context_provider
        if process_rules is not None and process_rules.enabled:
            if self.demo_mode or self.observation_shape is None:
                raise ValueError('Enabled process rules require live action and observation spaces')
            if (process_rules.action_shape != self.action_shape
                    or process_rules.observation_shape != self.observation_shape):
                raise ValueError('Process variable dictionary shapes do not match live spaces')

    def _response(self, status, level, rule_id, intervention, reason, task_id=None):
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'approved': status == 'Pass', 'status': status, 'level': level,
            'rule_id': rule_id, 'intervention': intervention,
            'reason': reason, 'demo_mode': self.demo_mode,
            'task_id': task_id,
        }
        if self.process_rules is not None and self.process_rules.enabled:
            result.update(rule_version=self.process_rules.version, environment=self.process_rules.environment)
        self.logs.append(result)
        print(f'[MonitoringAgent] {task_id or "-"} | Level {level} | {status} | {reason}')
        return result

    def _process_review(self, state, action, task_id, process_context):
        if self.process_rules is None or not self.process_rules.enabled:
            return None
        try:
            # An installed provider is authoritative; requests cannot override its context.
            if self.process_context_provider is not None:
                process_context = self.process_context_provider(deepcopy(state), deepcopy(action))
            violations = self.process_rules.evaluate(state, action, process_context)
        except Exception as exc:
            violations = [self.process_rules._violation(
                'R-03-CONTEXT', f'Process review failed: {type(exc).__name__}')]
        if violations:
            first = violations[0]
            result = self._response('Reject', 3, first['rule_id'], 'BLOCK', first['reason'], task_id)
            result.update({k: first[k] for k in ('basis', 'suggestion', 'rule_version', 'environment')})
            result['violations'] = violations
            return result
        return None

    def check_action(self, state, candidate_action, task_id=None, *, process_context=None):
        """R-01/R-02 precede optional, explicitly configured R-03 simulation fences."""
        if self.observation_shape is not None:
            error = self.observation_error(state)
            if error:
                return self._response('Reject', 3, 'STATE-INVALID', 'BLOCK', error, task_id)
        action, error = validate_action(candidate_action, self.action_shape)
        if error:
            return self._response('Reject', 3, 'R-01', 'BLOCK', error, task_id)
        error = check_bounds(action, self.action_min, self.action_max)
        if error:
            return self._response('Reject', 3, 'R-02', 'BLOCK', error, task_id)
        rejected = self._process_review(state, action, task_id, process_context)
        if rejected is not None:
            return rejected
        process_enabled = self.process_rules is not None and self.process_rules.enabled
        return self._response('Pass', 0, None, 'ALLOW',
                              ('所有已启用的规则检查通过（含配置的 R-03 仿真围栏）' if process_enabled
                               else '所有已启用的规则检查通过（R-03 未启用）'), task_id)

    def observation_error(self, observation):
        """Data integrity only; no invented physical thresholds or R-03 model."""
        return observation_integrity_error(observation, self.observation_shape)

    def check_observation(self, observation, task_id=None, *, process_context=None):
        error = self.observation_error(observation)
        if not error:
            rejected = self._process_review(observation, None, task_id, process_context)
            if rejected is not None:
                return rejected
        return self._response('Reject' if error else 'Pass', 3 if error else 0,
                              'STATE-INVALID' if error else None,
                              'BLOCK' if error else 'ALLOW',
                              error or ('观测完整性与已配置状态围栏检查通过'
                                        if self.process_rules is not None and self.process_rules.enabled
                                        else '过程观测数据完整性检查通过（非工艺安全判定）'), task_id)

    def check_request(self, action_request, observation=None):
        """A 的统一入口：ActionRequest={'task_id':str,'action':数组}。

        返回 SafetyDecision，五个固定字段：task_id、decision、rule_id、reason、risk_level。
        格式错误一律拒绝（fail closed），不把非法动作交给 B。
        """
        if not isinstance(action_request, dict):
            result = self._response('Reject', 3, 'R-01', 'BLOCK',
                                    'ActionRequest 必须为字典')
            task_id = None
        else:
            task_id = action_request.get('task_id')
            if not isinstance(task_id, str) or not task_id.strip():
                result = self._response('Reject', 3, 'R-01', 'BLOCK',
                                        '缺少有效 task_id', task_id=None)
                task_id = None
            else:
                result = self.check_action(observation, action_request.get('action'), task_id,
                                           process_context=action_request.get('safety_context'))
        identity = {k: action_request.get(k) for k in ('request_id', 'subtask_id')} if isinstance(action_request, dict) else {}
        result.update(identity)
        return {
            **identity,
            'task_id': task_id,
            'decision': 'pass' if result['approved'] else 'reject',
            'rule_id': result['rule_id'],
            'reason': result['reason'],
            'risk_level': result['level'],
            'demo_mode': self.demo_mode,
            **{k: deepcopy(result[k]) for k in ('basis', 'suggestion', 'rule_version', 'environment', 'violations') if k in result},
        }

    def get_logs(self):
        return deepcopy(self.logs)

    def clear_logs(self):
        self.logs.clear()

    def get_statistics(self):
        total = len(self.logs)
        passed = sum(x['approved'] for x in self.logs)
        rejected = total - passed
        return {'total': total, 'passed': passed, 'rejected': rejected,
                'warnings': sum(x['status'] == 'Warning' for x in self.logs),
                'rejection_rate': rejected / total if total else 0.0}

    def export_logs(self, path):
        """按 JSONL 写出可追踪日志，包含 task_id 和规则命中情况。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for record in self.logs:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


def safe_step(env, monitor, state, candidate_action):
    """兼容旧版：仅在检查通过时调用 env.step。"""
    result = monitor.check_action(state, candidate_action)
    if not result['approved']:
        return {'executed': False, 'monitoring_result': result, 'env_result': None}
    return {'executed': True, 'monitoring_result': result,
            'env_result': env.step(np.asarray(candidate_action, dtype=float))}


def safe_step_request(env, monitor, action_request, observation=None):
    """A/B 联调入口；reject 时绝不调用 env.step()，返回拒绝原因给 A。"""
    decision = monitor.check_request(action_request, observation)
    if decision['decision'] != 'pass':
        return {'executed': False, 'safety_decision': decision, 'env_result': None}
    action = np.asarray(action_request['action'], dtype=float)
    return {'executed': True, 'safety_decision': decision,
            'env_result': env.step(action)}
