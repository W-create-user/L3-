from dataclasses import replace
from l3_system.decision_agent import DecisionAgent
from l3_system.tests import test_goal_criteria as fixtures
import unittest


class HTNMethodsTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.GoalCriteriaTests()
        fixture.setUp()
        self.fixture = fixture
        self.capabilities = {'decision', 'monitoring', 'environment_execution'}

    def plan(self, value=3, samples=1, **changes):
        f = self.fixture
        records = [f.sample(i, value) for i in range(1, samples+1)]
        return DecisionAgent().create_goal_plan(f.goal, f.registry, records[-1], records[:-1],
                    capabilities=self.capabilities, **changes)

    def test_three_finite_methods_and_different_paths(self):
        adjust, hold, achieved = self.plan(), self.plan(1.5), self.plan(1.5, 3)
        self.assertEqual([adjust.method, hold.method, achieved.method],
                         ['adjust_target', 'hold_target', 'report_achieved'])
        self.assertNotEqual([t.task_id for t in hold.tasks], [t.task_id for t in adjust.tasks])
        self.assertFalse(any(t.capability == 'environment_execution' for t in achieved.tasks))

    def test_unknown_state_budget_and_missing_capability_stop_planning(self):
        for value in [float('nan'), 6]:
            with self.assertRaises(ValueError): self.plan(value)
        self.capabilities.remove('monitoring')
        with self.assertRaises(ValueError): self.plan()

    def test_cycle_and_missing_dependency(self):
        for dependency in ['missing', 'verify_result']:
            plan = self.plan()
            plan.tasks[0] = replace(plan.tasks[0], dependencies=(dependency,))
            with self.assertRaises(ValueError): plan.validate()

    def test_predicates_evaluated_and_effects_require_completion_evidence(self):
        plan = self.plan()
        self.assertEqual(plan.ready_tasks({}), [])
        self.assertEqual(plan.ready_tasks({'observation_valid':False}), [])
        facts = {'observation_valid':True}
        self.assertEqual([t.task_id for t in plan.ready_tasks(facts)], ['evaluate_goal'])
        with self.assertRaises(ValueError): plan.verify_completion('evaluate_goal', facts)
        facts['goal_evaluated'] = True
        self.assertEqual(plan.verify_completion('evaluate_goal', facts), {'goal_evaluated':True})
        with self.assertRaises(ValueError): plan.ready_tasks({'observation_valid':'true'})

    def test_execution_still_requires_safety_and_resource_facts(self):
        plan = self.plan(parent_plan_id='old-plan')
        completed = [t.task_id for t in plan.tasks[:4]]
        self.assertEqual(plan.parent_plan_id, 'old-plan')
        self.assertEqual(plan.ready_tasks({'safety_passed':True}, completed), [])
        self.assertEqual([t.task_id for t in plan.ready_tasks({'safety_passed':True,'resource_available':True}, completed)], ['execute_approved_action'])
