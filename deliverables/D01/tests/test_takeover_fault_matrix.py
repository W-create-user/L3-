"""Protocol fault matrix: FakeAdapter + explicit verdict stub, not SMPL/model efficacy."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.monitoring.statistical_monitor import StatisticalMonitoringAgent

class TakeoverFaultMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.adapter=FakeAdapter();self.executor=ExecutionAgent(self.adapter,Path(self.tmp.name)/'B.jsonl')
        self.executor.reset(seed=1101)
        # Normal verdict stub isolates orchestration; learned scoring is tested elsewhere.
        detector=SimpleNamespace(dimension=39,window_size=4,threshold=.5,label_scope='synthetic_fixture',
            score=lambda window:{'anomalous':False,'anomaly_score':.1})
        self.stale=False
        def provider():
            window=np.zeros((4,39));window[-1]=self.adapter.get_observation()
            if self.stale:window[-1]+=1
            return window
        self.monitor=StatisticalMonitoringAgent(self.adapter.action_space,self.adapter.observation_space,
            detector=detector,window_provider=provider,enable_experimental=True)
        self.system=MultiAgentSystem(executor=self.executor,monitor=self.monitor)
    def takeover(self):
        self.system.run_cycle(action_generator=lambda observation,space,attempt:[2.]*4)
        self.assertEqual(self.system.state,'takeover_requested');self.assertEqual(self.adapter.step_calls,0)
    def test_stale_window_blocks_ack_then_fresh_review_allows_resume_only(self):
        self.takeover();self.stale=True
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='ack cannot cure staleness')
        self.assertEqual(self.system.state,'takeover_requested');self.assertEqual(self.adapter.step_calls,0)
        self.stale=False;self.system.resume(operator='fixture',reason='fresh fixture')
        self.assertEqual(self.adapter.step_calls,0)
    def test_unreleased_assignment_blocks_resume(self):
        self.takeover();self.system.scheduler.active_agent_tasks['fixture']='unresolved'
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='lock still held')
        self.assertEqual(self.adapter.step_calls,0)
        self.system.scheduler.active_agent_tasks.clear()
        self.system.resume(operator='fixture',reason='fixture assignment resolved')
        self.assertEqual(self.adapter.step_calls,0)
    def test_duplicate_confirmation_does_not_execute(self):
        self.takeover();self.system.resume(operator='fixture',reason='first confirmation')
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='duplicate click')
        self.assertEqual(self.adapter.step_calls,0)
    def test_stale_prepared_request_cannot_use_old_approval(self):
        request=self.executor.prepare_action({'task_id':'stale-fixture','task_type':'control_cycle'},action=[0.]*4)
        decision=self.monitor.check_request(request,self.adapter.get_observation())
        self.adapter.state_version+=1  # Simulate an external state revision, not a physical step.
        with self.assertRaises(ValueError):self.executor.execute_action(request,decision)
        self.assertEqual(self.adapter.step_calls,0)
        self.executor.cancel_action_request(request['request_id'],'stale fixture disposed')
        self.assertFalse(self.executor._pending)
    def test_nan_observation_and_ended_episode_block_resume(self):
        self.takeover();self.adapter._obs[0]=float('nan')
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='invalid observation')
        self.adapter._obs[0]=0;self.adapter.done=True
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='episode ended')
        self.assertEqual(self.adapter.step_calls,0)
    def test_resume_does_not_bypass_new_action_gate_and_transitions_are_timestamped(self):
        self.takeover();self.system.resume(operator='fixture',reason='protocol review')
        self.assertEqual(self.adapter.step_calls,0)
        self.takeover()
        times=[datetime.fromisoformat(t['timestamp']) for t in self.system.transitions]
        self.assertTrue(all(t.tzinfo is not None for t in times));self.assertEqual(times,sorted(times))

if __name__=='__main__':unittest.main()
