import tempfile
import unittest
from pathlib import Path
from agents.monitoring_agent import MonitoringAgent
from runtime.cluster import AgentCluster
from tests.fixtures import TestAdapter


class SupplementTests(unittest.TestCase):
    def test_documented_state_rules_block_and_fail_closed(self):
        monitor = MonitoringAgent(TestAdapter().action_space, [
            {'index': 0, 'low': -1, 'high': 1, 'source': 'unit-test fixture only'}])
        self.assertFalse(monitor.check_action([2, 0], [0, 0])['approved'])
        self.assertFalse(monitor.check_action(None, [0, 0])['approved'])
        self.assertTrue(monitor.check_action([0, 0], [0, 0])['approved'])

    def test_rules_require_source(self):
        with self.assertRaises(ValueError):
            MonitoringAgent(TestAdapter().action_space, [{'index': 0, 'low': -1, 'high': 1}])

    def test_batch_and_human_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            cluster = AgentCluster(TestAdapter(), log_path=Path(folder)/'events.jsonl')
            self.assertEqual(len(cluster.run_batch(['检查状态', '执行一次控制任务'])), 2)
            self.assertFalse(cluster.run('执行一次控制任务', action=[2., 0.])['success'])
            with self.assertRaises(RuntimeError):
                cluster.run('执行一次控制任务')
            cluster.supervision.confirm_recovery('test_operator', 'fixture reviewed')
            self.assertTrue(cluster.run('执行一次控制任务')['success'])
            self.assertTrue(cluster.history[-1]['task']['subtasks'][0]['completion_criteria'])
