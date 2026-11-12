"""Synthetic A02 input contracts; no process-goal achievement claims."""
import copy
import json
from pathlib import Path
import unittest

from l3_system.planning.goals import validate_goal_spec, validate_registry


ROOT = Path(__file__).resolve().parents[2] / "configs" / "goals"


class GoalSpecTests(unittest.TestCase):
    def setUp(self):
        self.goal = json.loads((ROOT / "contract_fixture.json").read_text())
        self.registry = json.loads((ROOT / "contract_fixture_registry.json").read_text())

    def check(self):
        return validate_goal_spec(self.goal, self.registry)

    def test_valid_fixture_is_not_a_success_report(self):
        result = self.check()
        self.assertEqual(result, self.goal)
        self.assertNotIn("success", result)
        self.assertNotIn("approved", result)

    def test_input_and_output_do_not_alias(self):
        before = copy.deepcopy((self.goal, self.registry))
        result = self.check()
        result['target_range']['lower'] = -999
        self.assertEqual((self.goal, self.registry), before)
        registry = validate_registry(self.registry)
        registry['variables'].clear()
        self.assertTrue(self.registry['variables'])

    def test_missing_and_unknown_goal_fields(self):
        original = copy.deepcopy(self.goal)
        for key in original:
            with self.subTest(missing=key):
                self.goal = copy.deepcopy(original); del self.goal[key]
                with self.assertRaises(ValueError): self.check()
        self.goal = dict(original, reward_target=1)
        with self.assertRaises(ValueError): self.check()

    def test_wrong_version_and_environment(self):
        for key, value in [('schema_version','1'),('environment_id','MAbEnvGym'),('registry_version','other')]:
            with self.subTest(key=key):
                old = self.goal[key]; self.goal[key] = value
                with self.assertRaises(ValueError): self.check()
                self.goal[key] = old

    def test_unknown_or_action_variable(self):
        for name in ['missing', 'test_feed']:
            self.goal['variable'] = name
            with self.assertRaises(ValueError): self.check()

    def test_units_and_coordinates_are_not_converted(self):
        for key in ['unit','coordinate']:
            old = self.goal[key]; self.goal[key] = 'other'
            with self.assertRaises(ValueError): self.check()
            self.goal[key] = old
            action = self.goal['allowed_actions']['test_feed']
            old = action[key]; action[key] = 'other'
            with self.assertRaises(ValueError): self.check()
            action[key] = old

    def test_invalid_budgets(self):
        for key in ['max_steps','hold_steps']:
            old = self.goal[key]
            for value in [True, False, 0, -1, 1.5, '3', None]:
                with self.subTest(key=key, value=value):
                    self.goal[key] = value
                    with self.assertRaises(ValueError): self.check()
            self.goal[key] = old
        self.goal['hold_steps'] = 11
        with self.assertRaises(ValueError): self.check()

    def test_nonfinite_and_non_numeric_ranges(self):
        for interval in [self.goal['target_range'], self.goal['allowed_actions']['test_feed']['range']]:
            old = interval['lower']
            for value in [True, '0', None, float('nan'), float('inf'), -float('inf'), 10**1000]:
                with self.subTest(value=str(value)[:30]):
                    interval['lower'] = value
                    with self.assertRaises(ValueError): self.check()
            interval['lower'] = old

    def test_reversed_and_out_of_domain_target(self):
        for interval in [{'lower':3,'upper':2}, {'lower':-1,'upper':2}, {'lower':1,'upper':6}]:
            self.goal['target_range'] = interval
            with self.assertRaises(ValueError): self.check()

    def test_equal_and_inclusive_bounds(self):
        self.goal['target_range'] = {'lower':0,'upper':5}
        self.goal['allowed_actions']['test_feed']['range'] = {'lower':-1,'upper':1}
        self.check()
        self.goal['target_range'] = {'lower':2,'upper':2}
        self.goal['hold_steps'] = self.goal['max_steps']
        self.check()

    def test_actions_cannot_be_missing_extra_or_widened(self):
        original = copy.deepcopy(self.goal['allowed_actions'])
        for actions in [{}, dict(original, fake=original['test_feed'])]:
            self.goal['allowed_actions'] = actions
            with self.assertRaises(ValueError): self.check()
        self.goal['allowed_actions'] = original
        for interval in [{'lower':-2,'upper':0}, {'lower':0,'upper':2}, {'lower':1,'upper':-1}]:
            original['test_feed']['range'] = interval
            with self.assertRaises(ValueError): self.check()

    def test_nested_unknown_fields_rejected(self):
        for obj in [self.goal['target_range'], self.goal['allowed_actions']['test_feed'],
                    self.goal['allowed_actions']['test_feed']['range']]:
            obj['typo'] = 1
            with self.assertRaises(ValueError): self.check()
            del obj['typo']

    def test_registry_requires_evidence_and_exact_fields(self):
        self.registry['evidence_ref'] = ''
        with self.assertRaises(ValueError): self.check()
        self.registry['evidence_ref'] = 'fixture'
        self.registry['unexpected'] = True
        with self.assertRaises(ValueError): self.check()

    def test_registry_duplicate_or_out_of_bounds_indices(self):
        v = self.registry['variables']
        v['duplicate'] = copy.deepcopy(v['test_concentration'])
        with self.assertRaises(ValueError): self.check()
        del v['duplicate']
        for value in [True, -1, 1, '0']:
            v['test_concentration']['index'] = value
            with self.assertRaises(ValueError): self.check()

    def test_registry_must_cover_all_action_axes(self):
        self.registry['action_size'] = 2
        with self.assertRaises(ValueError): self.check()

    def test_registry_unresolved_metadata_is_rejected(self):
        v = self.registry['variables']['test_concentration']
        for key in ['unit','coordinate']:
            old = v[key]
            for value in ['unconfirmed','unknown','pending','TBD','',None]:
                v[key] = value
                with self.assertRaises(ValueError): self.check()
            v[key] = old

    def test_registry_action_bounds_required(self):
        self.registry['variables']['test_feed']['bounds'] = None
        with self.assertRaises(ValueError): self.check()

    def test_no_observation_bounds_invents_no_limits(self):
        self.registry['variables']['test_concentration']['bounds'] = None
        self.goal['target_range'] = {'lower':-1,'upper':100}
        self.assertEqual(self.check()['target_range'], self.goal['target_range'])

    def test_malformed_objects_rejected(self):
        for value in [None, [], '', 1]:
            with self.assertRaises(ValueError): validate_goal_spec(value, self.registry)
            with self.assertRaises(ValueError): validate_goal_spec(self.goal, value)

    def test_identifiers_not_silently_normalized(self):
        for value in ['', '   ', ' GOAL ', None]:
            self.goal['goal_id'] = value
            with self.assertRaises(ValueError): self.check()


if __name__ == '__main__':
    unittest.main()
