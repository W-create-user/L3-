"""Synthetic fixtures only; these limits are not Atropine/mAb physical limits."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from l3_system.agents.monitoring_agent import MonitoringAgent, safe_step_request
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.execution import ExecutionAgent
from l3_system.main import run_control_cycle
from l3_system.runtime import MultiAgentSystem
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter


def fixture_config():
    basis = dict(kind='team_experiment', reference='test_process_rules.py synthetic fixture',
                 reviewed_by='automated-test-fixture-not-a-human-approval', reviewed_on='2026-11-12')
    def rule(name, kind, priority, **kwargs):
        return dict(rule_id='R-03-'+name, kind=kind, enabled=True, priority=priority,
                    basis=deepcopy(basis), suggestion='Restore fixture state and re-review', **kwargs)
    return dict(schema_version='ProcessRules/1.0', enabled=True, environment='SYNTHETIC_TEST_ONLY',
                version='fixture-v1', observation_shape=[39], action_shape=[4],
                variables={'sensor': dict(source='observation', index=0, unit='fixture-unit', coordinate='fixture-absolute'),
                           'ready': dict(source='observation', index=1, unit='fixture-flag', coordinate='fixture-absolute'),
                           'control': dict(source='action', index=0, unit='fixture-unit', coordinate='fixture-delta')},
                rules=[rule('STATE', 'range', 2, variable='sensor', lower=-0.5, upper=0.5),
                       rule('READY', 'precondition', 1, variable='ready', equals=0),
                       rule('RATE', 'rate', 3, variable='control', max_rate_per_second=0.2),
                       rule('AGE', 'freshness', 0, max_age_seconds=2)])


def fixture_context():
    return dict(environment='SYNTHETIC_TEST_ONLY', episode_id='fixture-episode',
                variables={name: {k: v[k] for k in ('unit', 'coordinate')}
                           for name, v in fixture_config()['variables'].items()},
                reviewed_at=10.0, observed_at=9.0, action_at=10.0, previous_action_at=9.0,
                previous_action=[0.0]*4, previous_action_episode_id='fixture-episode')


class ProcessRuleTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.context = fixture_context()
        self.observation = [0.0]*39
        self.action = [0.0]*4

    def engine(self):
        return ProcessRuleEngine(self.config, allow_experimental=True)

    def result(self):
        return self.engine().evaluate(self.observation, self.action, self.context)

    def bad_config(self):
        with self.assertRaises(ValueError):
            self.engine()

    def test_normal_and_inclusive_boundaries(self):
        for value in (-0.5, 0.0, 0.5):
            self.observation[0] = value
            self.assertEqual(self.result(), [])

    def test_both_sides_outside_range(self):
        for value in (-0.50001, 0.50001):
            self.observation[0] = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-STATE')

    def test_rate_boundary_and_both_directions(self):
        for value in (-0.2, 0.2):
            self.action[0] = value
            self.assertEqual(self.result(), [])
        for value in (-0.21, 0.21):
            self.action[0] = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-RATE')

    def test_rate_uses_elapsed_time(self):
        self.action[0] = 0.3
        self.context['previous_action_at'] = 8.0
        self.assertEqual(self.result(), [])

    def test_rate_requires_positive_time(self):
        for value in (10, 11, None, float('nan'), True):
            self.context['previous_action_at'] = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_rate_rejects_previous_episode(self):
        self.context['previous_action_episode_id'] = 'old-episode'
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_previous_action_required_finite_and_shape(self):
        for value in (None, [0], [float('inf')]*4, ['0']*4):
            self.context['previous_action'] = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_precondition(self):
        self.observation[1] = 1
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-READY')

    def test_freshness_boundary_and_future_clock(self):
        self.context['observed_at'] = 8.0
        self.assertEqual(self.result(), [])
        self.context['observed_at'] = 7.9
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-AGE')
        self.context['observed_at'] = 11.0
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_missing_context_environment_and_episode(self):
        for value in (None, {}, dict(self.context, environment='mAb'), dict(self.context, episode_id='')):
            self.context = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')
            self.context = fixture_context()

    def test_units_and_coordinates_required(self):
        for field in ('unit', 'coordinate'):
            self.context['variables']['sensor'][field] = 'wrong'
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')
            self.context = fixture_context()
        del self.context['variables']
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_invalid_state_and_action(self):
        for value in (None, [0], [float('nan')]*39, ['0']*39):
            self.observation = value
            self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')
        self.observation = [0.0]*39
        self.action = [float('inf')]*4
        self.assertEqual(self.result()[0]['rule_id'], 'R-03-CONTEXT')

    def test_deterministic_priority_with_all_violations(self):
        self.observation[0] = 1; self.observation[1] = 1; self.action[0] = 0.3
        self.context['observed_at'] = 0
        violations = self.result()
        self.assertEqual([v['rule_id'] for v in violations], ['R-03-AGE','R-03-READY','R-03-STATE','R-03-RATE'])
        for item in violations:
            self.assertTrue(all(k in item for k in ('basis','suggestion','risk_level','rule_version','environment')))

    def test_disabled_rule_does_not_require_metadata(self):
        self.config['rules'][2]['enabled'] = False
        del self.context['previous_action']
        del self.context['variables']['control']
        self.assertEqual(self.result(), [])

    def test_disabled_engine_does_not_claim_process_check(self):
        self.config['enabled'] = False
        self.assertEqual(self.engine().evaluate(None, None, None), [])

    def test_config_copied_before_external_mutation(self):
        engine = self.engine()
        self.config['rules'][0]['upper'] = 99
        self.observation[0] = 1
        self.assertEqual(engine.evaluate(self.observation,self.action,self.context)[0]['rule_id'], 'R-03-STATE')

    def test_config_conflicting_ranges(self):
        rule = deepcopy(self.config['rules'][0]); rule.update(rule_id='R-03-OTHER',lower=2,upper=3)
        self.config['rules'].append(rule); self.bad_config()

    def test_config_conflicting_preconditions(self):
        rule = deepcopy(self.config['rules'][1]); rule.update(rule_id='R-03-OTHER',equals=1)
        self.config['rules'].append(rule); self.bad_config()

    def test_config_range_precondition_conflict(self):
        self.config['rules'][1].update(variable='sensor',equals=2); self.bad_config()

    def test_config_nonfinite_limits_and_boolean_enable(self):
        self.config['rules'][0]['upper'] = float('inf'); self.bad_config()
        self.config = fixture_config(); self.config['enabled'] = 'false'; self.bad_config()

    def test_config_missing_basis_and_unknown_variable(self):
        self.config['rules'][0]['basis']['reference'] = ''; self.bad_config()
        self.config = fixture_config(); self.config['rules'][0]['variable'] = 'missing'; self.bad_config()

    def test_config_duplicate_rule_and_variable_index(self):
        self.config['rules'].append(deepcopy(self.config['rules'][0])); self.bad_config()
        self.config = fixture_config(); self.config['variables']['sensor']['index'] = 39; self.bad_config()

    def test_team_thresholds_need_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            ProcessRuleEngine(self.config)

    def test_enabled_empty_rules_rejected(self):
        self.config['rules'] = []; self.bad_config()

    def test_rate_cannot_target_observation(self):
        self.config['rules'][2]['variable'] = 'sensor'; self.bad_config()

    def test_json_nonfinite_constants_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'; path.write_text(json.dumps(dict(self.config,version=float('nan'))))
            with self.assertRaises(ValueError):
                ProcessRuleEngine.from_json(path,allow_experimental=True)


class ProcessIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.adapter=FakeAdapter()
        self.context=fixture_context()
        self.monitor=MonitoringAgent(self.adapter.action_space,self.adapter.observation_space,
                                    process_rules=ProcessRuleEngine(fixture_config(),allow_experimental=True),
                                    process_context_provider=lambda state, action: deepcopy(self.context))
        self.executor=ExecutionAgent(self.adapter,Path(self.tmp.name)/'B.jsonl')

    def test_normal_full_chain_executes_one_step(self):
        result=run_control_cycle(goal='执行一次控制任务',executor=self.executor,monitor=self.monitor)
        self.assertEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.adapter.step_calls,1)
        self.assertTrue(result['summary']['r03_enabled'])

    def test_process_reject_stops_full_chain_before_step(self):
        self.adapter._obs[0]=1
        result=run_control_cycle(goal='执行一次控制任务',executor=self.executor,monitor=self.monitor)
        self.assertNotEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.adapter.step_calls,0)
        self.assertFalse(self.executor._pending)

    def test_final_gate_rechecks_changed_context(self):
        count=0
        def changing(state, action):
            nonlocal count
            count+=1
            return dict(self.context,observed_at=0 if count>1 else 9)
        self.monitor.process_context_provider=changing
        result=run_control_cycle(goal='执行一次控制任务',executor=self.executor,monitor=self.monitor)
        self.assertNotEqual(result['summary']['run_status'],'completed')
        self.assertEqual(count,2)
        self.assertEqual(self.adapter.step_calls,0)

    def test_existing_R02_has_priority(self):
        result=self.monitor.check_request({'task_id':'T','action':[2]*4},self.adapter.get_observation())
        self.assertEqual(result['rule_id'],'R-02')

    def test_provider_exception_blocks(self):
        def fail(*args): raise RuntimeError('missing sensor clock')
        self.monitor.process_context_provider=fail
        result=safe_step_request(self.adapter,self.monitor,{'task_id':'T','action':[0]*4},self.adapter.get_observation())
        self.assertFalse(result['executed']); self.assertEqual(self.adapter.step_calls,0)

    def test_request_metadata_cannot_override_provider(self):
        self.context['observed_at']=0
        result=self.monitor.check_request({'task_id':'T','action':[0]*4,'safety_context':fixture_context()},self.adapter.get_observation())
        self.assertEqual(result['rule_id'],'R-03-AGE')
        self.assertIn('basis',result); self.assertIn('suggestion',result)

    def test_request_context_supported_without_provider(self):
        self.monitor.process_context_provider=None
        result=self.monitor.check_request({'task_id':'T','action':[0]*4,'safety_context':fixture_context()},self.adapter.get_observation())
        self.assertEqual(result['decision'],'pass')
        result=self.monitor.check_request({'task_id':'T','action':[0]*4},self.adapter.get_observation())
        self.assertEqual(result['decision'],'reject')

    def test_recovery_confirmation_cannot_skip_process_fence(self):
        system=MultiAgentSystem(executor=self.executor,monitor=self.monitor)
        self.context['observed_at']=0
        system.run_cycle()
        self.assertEqual(system.state,'takeover_requested')
        with self.assertRaises(RuntimeError):system.resume(operator='fixture',reason='confirm only')
        self.assertEqual(system.state,'takeover_requested'); self.assertEqual(self.adapter.step_calls,0)
        self.context['observed_at']=9
        system.resume(operator='fixture',reason='fresh context restored')
        self.assertEqual(system.state,'autonomous'); self.assertEqual(self.adapter.step_calls,0)
        self.assertEqual(system.run_cycle()['summary']['run_status'],'completed')
        self.assertEqual(self.adapter.step_calls,1)

    def test_constructor_refuses_shapes_without_live_space(self):
        engine=ProcessRuleEngine(fixture_config(),allow_experimental=True)
        with self.assertRaises(ValueError):MonitoringAgent(self.adapter.action_space,process_rules=engine)
        config=fixture_config(); config['observation_shape']=[40]
        with self.assertRaises(ValueError):
            MonitoringAgent(self.adapter.action_space,self.adapter.observation_space,
                            process_rules=ProcessRuleEngine(config,allow_experimental=True))

    def test_exported_rejection_retains_provenance(self):
        self.context['observed_at']=0
        self.monitor.check_request({'task_id':'T','request_id':'R','action':[0]*4},self.adapter.get_observation())
        path=Path(self.tmp.name)/'D.jsonl'; self.monitor.export_logs(path)
        log=json.loads(path.read_text()); self.assertEqual(log['request_id'],'R')
        self.assertEqual(log['rule_version'],'fixture-v1'); self.assertIn('basis',log)

    def test_returned_reviews_cannot_rewrite_nested_log_basis(self):
        self.context['observed_at']=0
        review=self.monitor.check_request({'task_id':'T','action':[0]*4},self.adapter.get_observation())
        review['basis']['reference']='tampered'
        log=self.monitor.get_logs()[0]
        self.assertNotEqual(log['basis']['reference'],'tampered')
        log['basis']['reference']='tampered-again'
        self.assertNotEqual(self.monitor.get_logs()[0]['basis']['reference'],'tampered-again')

    def test_disabled_repository_config_retains_original_behavior(self):
        config=Path(__file__).resolve().parents[2]/'configs/safety/atropine.disabled.json'
        # Repository root is one level above l3_system.
        engine=ProcessRuleEngine.from_json(config)
        monitor=MonitoringAgent(self.adapter.action_space,self.adapter.observation_space,process_rules=engine)
        result=monitor.check_request({'task_id':'T','action':[0]*4},self.adapter.get_observation())
        self.assertEqual(result['decision'],'pass')
        self.assertNotIn('rule_version',result)


if __name__ == '__main__':
    unittest.main()
