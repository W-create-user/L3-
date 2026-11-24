"""D/B contract smoke; synthetic fixture assertions, zero simulator steps.

Not a model or process-safety validation. Run from the repository root.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.monitoring.risk_policy import RiskPolicy, RiskEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    def expect(name, result, expected):
        assert result == expected, (name, result, expected)
        rows.append({'case': name, 'expected': expected, 'actual': result, 'passed': True})
    # Bounds deliberately synthetic. Live users must supply B adapter spaces.
    for name, n, m in [('Atropine', 39, 4), ('mAb', 1970, 9)]:
        action_space = SimpleNamespace(shape=(m,), low=np.full(m, -1.), high=np.full(m, 1.))
        observation_space = SimpleNamespace(shape=(n,))
        monitor = MonitoringAgent(action_space, observation_space)
        request = {'task_id': 'fixture', 'request_id': name+'-1', 'subtask_id': 'control', 'action': [0.]*m}
        with contextlib.redirect_stdout(io.StringIO()):
            decision = monitor.check_request(request, [0.]*n)
            expect(name+'.valid_action', decision['decision'], 'pass')
            expect(name+'.identity', decision['request_id'], request['request_id'])
            expect(name+'.bad_shape', monitor.check_request({**request, 'action': [0.]}, [0.]*n)['rule_id'], 'R-01')
            expect(name+'.out_of_bounds', monitor.check_request({**request, 'action': [2.]*m}, [0.]*n)['rule_id'], 'R-02')
            expect(name+'.missing_state', monitor.check_request(request, None)['rule_id'], 'STATE-INVALID')
    config_path = Path(__file__).resolve().parents[3]/'configs/safety/atropine_experiment_20261116.json'
    config = json.loads(config_path.read_text())
    rules = ProcessRuleEngine(config, allow_experimental=True)
    variables = {k: {x:v[x] for x in ('unit','coordinate')} for k,v in config['variables'].items()}
    context = {'environment':'Atropine','episode_id':'fixture','variables':variables}
    state = np.zeros(39); state[5] = 10.; state[6] = 11.
    expect('Atropine.relative_upper_equality', bool(rules.evaluate(state, np.zeros(4), context)), False)
    state[6] = 11.01
    expect('Atropine.relative_upper_breach', rules.evaluate(state, np.zeros(4), context)[0]['rule_id'], 'R-03-ATROPINE-EXPERIMENT-E')
    expect('Atropine.missing_context', rules.evaluate(state, np.zeros(4), {})[0]['rule_id'], 'R-03-CONTEXT')
    policy = RiskPolicy(); policy.reset('fixture')
    for version in range(1,4):
        # Explicitly fixture evidence: no real model output is claimed.
        risk = policy.evaluate(RiskEvidence('fixture', version, True, 'valid', True, False, 'in_scope'))
    expect('D05.history_without_human_review', risk['automatic_execution_blocked'], True)
    risk = policy.confirm_recovery('fixture', state_version=3, reviewer_ref='fixture-review-only')
    expect('D05.review_allows_gate_request', risk['risk_level'], 1)
    expect('D05.never_authorizes_step', risk['execution_authorized'], False)
    risk = policy.evaluate(RiskEvidence('fixture', 4, True, 'valid', True, None, 'unknown'))
    expect('D05.unavailable_model', risk['mode'], 'takeover_requested')
    risk = policy.evaluate(RiskEvidence('fixture', 4, True, 'valid', True, False, 'in_scope'))
    expect('D05.repeated_version', risk['automatic_execution_blocked'], True)
    report = {'schema_version':'DBContractSmoke/1.0', 'fixture_only':True,
              'simulator_step_calls':0, 'model_inference_performed':False,
              'physical_safety_certified':False, 'cases':rows, 'all_passed':True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'checks':len(rows), 'all_passed':True, 'simulator_step_calls':0}))

if __name__ == '__main__':
    main()
