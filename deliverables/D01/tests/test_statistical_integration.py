"""End-to-end routing with a trained synthetic detector and counting FakeAdapter."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from train_isolation_forest import synthetic_episodes
from l3_system.monitoring.anomaly_detector import AnomalyDetector
from l3_system.monitoring.statistical_monitor import StatisticalMonitoringAgent
from l3_system.agents.monitoring_agent import safe_step
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem


class StatisticalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        episodes=synthetic_episodes()
        cls.model=AnomalyDetector();cls.model.fit(episodes,label_scope='synthetic_fixture')
        # Select a known passing routing fixture, not a performance evaluation sample.
        candidates=[]
        for episode in episodes[:6]:
            for i in range(4,len(episode['observations'])):
                window=np.asarray(episode['observations'][i-4:i])
                after=window.copy();after[-1]+=0.01
                if not cls.model.score(window)['anomalous'] and not cls.model.score(after)['anomalous']:
                    candidates.append(window)
                    break
        assert candidates,'synthetic healthy routing fixture unavailable'
        cls.healthy=candidates[0]

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.adapter=FakeAdapter()
        self.adapter._obs=self.healthy[-1].copy()
        # Fixed synthetic history is for routing tests, not a real temporal collector.
        def healthy_provider():
            window=self.healthy.copy();window[-1]=self.adapter.get_observation();return window
        self.provider=healthy_provider
        self.monitor=StatisticalMonitoringAgent(self.adapter.action_space,self.adapter.observation_space,
                detector=self.model,window_provider=lambda:self.provider(),enable_experimental=True)
        self.executor=ExecutionAgent(self.adapter,Path(self.tmp.name)/'B.jsonl')
        self.system=MultiAgentSystem(executor=self.executor,monitor=self.monitor)

    def test_healthy_full_chain_one_step(self):
        result=self.system.run_cycle()
        self.assertEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.adapter.step_calls,1)
        self.assertTrue(result['summary']['statistical_r03_enabled'])
        self.assertFalse(result['summary']['r03_enabled'])

    def test_anomaly_requests_takeover_and_cannot_resume_until_rechecked(self):
        self.adapter._obs[:]=100
        result=self.system.run_cycle()
        self.assertEqual(self.system.state,'takeover_requested')
        self.assertEqual(self.adapter.step_calls,0)
        self.assertFalse(self.system.scheduler.active_agent_tasks)
        self.assertFalse(self.executor._pending)
        with self.assertRaises(RuntimeError):self.system.resume(operator='fixture',reason='ack only')
        self.assertEqual(self.adapter.step_calls,0)
        self.adapter._obs=self.healthy[-1].copy()
        self.system.resume(operator='fixture',reason='synthetic context restored')
        self.assertEqual(self.adapter.step_calls,0)
        self.assertEqual(self.system.run_cycle()['summary']['run_status'],'completed')

    def test_legacy_entry_cannot_bypass_learned_gate(self):
        self.adapter._obs[:]=100
        result=safe_step(self.adapter,self.monitor,self.adapter.get_observation(),[0]*4)
        self.assertFalse(result['executed']);self.assertEqual(self.adapter.step_calls,0)

    def test_final_review_sees_window_change(self):
        calls=0
        def changing():
            nonlocal calls
            calls+=1
            window=self.healthy.copy();window[-1]=self.adapter.get_observation()
            return window if calls==1 else window+1
        self.provider=changing
        result=self.system.run_cycle()
        self.assertEqual(self.adapter.step_calls,0)
        self.assertEqual(self.system.state,'takeover_requested')

    def test_hard_reject_does_not_invoke_model(self):
        with patch.object(self.model,'score',side_effect=AssertionError('hard gate first')):
            result=self.monitor.check_request({'task_id':'T','action':[2]*4},self.adapter.get_observation())
        self.assertEqual(result['rule_id'],'R-02')

    def test_malformed_score_blocks_and_one_log_per_review(self):
        with patch.object(self.model,'score',return_value={'anomalous':'false'}):
            result=self.monitor.check_request({'task_id':'T','action':[0]*4},self.adapter.get_observation())
        self.assertEqual(result['decision'],'reject');self.assertEqual(len(self.monitor.logs),1)


if __name__=='__main__':unittest.main()
