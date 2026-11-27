from types import SimpleNamespace
import unittest
import numpy as np
from l3_system.execution.controllers.source_mpc import SourceMPCPolicy
from l3_system.tests.test_live_orchestration_contracts import FakeBox

class SourceMPCPolicyTests(unittest.TestCase):
    def setUp(self):
        self.space=FakeBox([0]*9,[1]*9)
        self.adapter=SimpleNamespace(metadata={'environment_id':'MAbEnvGym'},action_space=self.space)
    def test_explicit_research_opt_in_required(self):
        with self.assertRaises(RuntimeError):SourceMPCPolicy(self.adapter)
    def test_failed_solver_cannot_release_or_commit_action(self):
        policy=SourceMPCPolicy(self.adapter,allow_research=True)
        policy.solver=SimpleNamespace(stats={'status':'Infeasible_Problem_Detected'},fixvar=lambda *args:None,solve=lambda:None)
        with self.assertRaisesRegex(RuntimeError,'candidate action is not released'):policy.predict(np.zeros(1970),self.space)
        self.assertFalse(policy._pending)
        with self.assertRaises(ValueError):policy.commit(True)
