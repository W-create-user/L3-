import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from l3_system.execution import ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.planning.goal_runtime import run_numeric_goal
from l3_system.runtime import MultiAgentSystem
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter

class Policy:
    def __init__(self,action):self.action=action;self.commits=[];self.resets=0;self.pending=False
    def reset(self):self.resets+=1;self.pending=False
    def predict(self,obs,space):
        if self.pending:raise RuntimeError('Previous proposal not committed')
        self.pending=True;return self.action
    def commit(self,executed):self.commits.append(executed);self.pending=False

class StatefulGoalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.adapter=FakeAdapter()
        self.system=MultiAgentSystem(executor=ExecutionAgent(self.adapter,log_path=Path(self.tmp.name)/'B.jsonl'),monitor=MonitoringAgent(self.adapter.action_space,self.adapter.observation_space))
        root=Path(__file__).resolve().parents[2]/'configs/goals'
        self.registry=json.loads((root/'atropine_model_registry.json').read_text());self.goal=json.loads((root/'atropine_goal_regression.json').read_text());self.goal['target_range']={'lower':.01,'upper':.04}
    def test_physical_cycles_commit_but_read_only_achievement_does_not(self):
        policy=Policy([0]*4);result=run_numeric_goal(self.system,self.goal,self.registry,environment='AtropineEnvGym',policy=policy)
        self.assertEqual(result['summary']['status'],'succeeded');self.assertEqual(policy.commits,[True,True]);self.assertEqual(policy.resets,1)
    def test_rejected_goal_action_discards_policy_state_without_step(self):
        policy=Policy([.5]*4);result=run_numeric_goal(self.system,self.goal,self.registry,environment='AtropineEnvGym',policy=policy)
        self.assertEqual(result['summary']['status'],'failed');self.assertEqual(policy.commits,[False]);self.assertEqual(self.adapter.step_calls,0)
    def test_commit_error_after_step_latches_takeover(self):
        policy=Policy([0]*4)
        def broken(executed):raise RuntimeError('policy state persistence failure')
        policy.commit=broken
        with self.assertRaises(RuntimeError):run_numeric_goal(self.system,self.goal,self.registry,environment='AtropineEnvGym',policy=policy)
        self.assertEqual(self.adapter.step_calls,1);self.assertEqual(self.system.state,'takeover_requested')
