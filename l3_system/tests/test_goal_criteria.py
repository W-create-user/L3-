import copy
import json
from pathlib import Path
import unittest
from l3_system.planning.goals import evaluate_goal

ROOT = Path(__file__).resolve().parents[2]


class GoalCriteriaTests(unittest.TestCase):
    def setUp(self):
        self.goal = json.loads((ROOT/'configs/goals/contract_fixture.json').read_text())
        self.registry = json.loads((ROOT/'configs/goals/contract_fixture_registry.json').read_text())

    def sample(self, step, value=1.5, **changes):
        return dict(goal_id=self.goal['goal_id'], environment_id=self.goal['environment_id'], registry_version=self.goal['registry_version'],
                    episode_id='E1', step=step, observation=[value], safety_ok=True, **changes)

    def evaluate(self, values, start=1):
        samples = [self.sample(i, x) for i, x in enumerate(values, start)]
        return evaluate_goal(samples[-1], samples[:-1], self.goal, self.registry)

    def test_window_requires_actual_steps_not_reset(self):
        self.assertFalse(self.evaluate([1.5]*3, start=0)['achieved'])
        self.assertTrue(self.evaluate([1.5]*4, start=0)['achieved'])

    def test_window_resets_on_excursion(self):
        result = self.evaluate([1.5, 1.5, 3, 1.5])
        self.assertEqual(result['consecutive_steps'], 1)
        self.assertFalse(result['achieved'])

    def test_interval_endpoints_and_deviation(self):
        self.assertTrue(self.evaluate([1, 2, 1])['achieved'])
        self.assertEqual(self.evaluate([3])['deviation'], 1)
        self.assertEqual(self.evaluate([0.5])['deviation'], -0.5)

    def test_deadline_last_step_and_late_window(self):
        self.goal['max_steps'] = 3
        self.assertTrue(self.evaluate([1.5]*3)['achieved'])
        self.assertEqual(self.evaluate([3]*3)['status'], 'failed')
        self.assertEqual(self.evaluate([3, 1.5, 1.5, 1.5])['reason'], 'step_budget_exceeded')

    def test_invalid_identity_gap_duplicate_episode_shape_and_nan_fail_closed(self):
        for field, value in [('goal_id','wrong'), ('environment_id','wrong'), ('registry_version','wrong'),
                             ('episode_id','E2'), ('step',1), ('step',3),
                             ('observation',[]), ('observation',[float('nan')]), ('safety_ok',None)]:
            with self.subTest(field=field, value=value):
                current = self.sample(2)
                current[field] = value
                result = evaluate_goal(current, [self.sample(1)], self.goal, self.registry)
                self.assertEqual(result['status'], 'unevaluable')
                self.assertFalse(result['achieved'])

    def test_previous_safety_violation_prevents_success(self):
        samples = [self.sample(i) for i in (1,2,3)]
        samples[0]['safety_ok'] = False
        self.assertEqual(evaluate_goal(samples[-1], samples[:-1], self.goal, self.registry)['reason'], 'safety_violation')

    def test_missing_mapping_and_unit_conflict_raise(self):
        for field, value in [('variable','reward'), ('unit','wrong')]:
            goal = copy.deepcopy(self.goal)
            goal[field] = value
            with self.assertRaises(ValueError):
                evaluate_goal(self.sample(1), [], goal, self.registry)

    def test_domain_and_truncated_history_rejected(self):
        self.assertEqual(self.evaluate([6])['status'], 'unevaluable')
        self.assertEqual(self.evaluate([1.5], start=5)['status'], 'unevaluable')

    def test_b_registry_rebuild_matches_committed_output(self):
        from experiments.build_a_registry import build
        registry, notes = build()
        self.assertEqual(registry, json.loads((ROOT/'configs/goals/atropine_model_registry.json').read_text()))
        self.assertEqual(notes, json.loads((ROOT/'configs/goals/atropine_registry_provenance.json').read_text()))
        self.assertNotIn('estimator_x_1', registry['variables'])
        self.assertEqual(registry['variables']['E_factor_current']['index'], 6)
        self.assertFalse(notes['r03_enabled'])
