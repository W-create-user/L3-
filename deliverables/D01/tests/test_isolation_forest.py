from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from train_isolation_forest import synthetic_episodes
from l3_system.monitoring.anomaly_detector import AnomalyDetector,features
from l3_system.monitoring.statistical_monitor import StatisticalMonitoringAgent
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter


class ForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes=synthetic_episodes();cls.model=AnomalyDetector()
        cls.card=cls.model.fit(cls.episodes,label_scope='synthetic_fixture')

    def test_training_and_roundtrip(self):
        window=self.episodes[0]['observations'][:4]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model';self.model.save(path)
            restored=AnomalyDetector.load_trusted(path)
            self.assertEqual(self.model.score(window),restored.score(window))
            with (path/'model.pkl').open('ab') as stream:stream.write(b'tamper')
            with self.assertRaises(ValueError):AnomalyDetector.load_trusted(path)

    def test_no_leakage_from_test_values_to_threshold_or_scaler(self):
        episodes=deepcopy(self.episodes)
        for e in episodes:
            if e['split']=='test':e['observations']=(np.asarray(e['observations'])+100).tolist()
        other=AnomalyDetector();other.fit(episodes,label_scope='synthetic_fixture')
        self.assertEqual(self.model.threshold,other.threshold)
        np.testing.assert_array_equal(self.model.scaler.mean_,other.scaler.mean_)

    def test_group_crossing_rejected(self):
        episodes=deepcopy(self.episodes);episodes[-1]['lineage_group']=episodes[0]['lineage_group']
        with self.assertRaises(ValueError):AnomalyDetector().fit(episodes,label_scope='synthetic_fixture')

    def test_abnormal_train_rejected(self):
        episodes=deepcopy(self.episodes);episodes[0]['labels'][0]=True
        with self.assertRaises(ValueError):AnomalyDetector().fit(episodes,label_scope='synthetic_fixture')

    def test_scope_mixing_rejected(self):
        episodes=deepcopy(self.episodes);episodes[-1]['label_scope']='process_anomaly'
        with self.assertRaises(ValueError):AnomalyDetector().fit(episodes,label_scope='synthetic_fixture')

    def test_feature_finite_and_shape(self):
        self.assertEqual(features(np.zeros((4,39))).shape,(234,))
        with self.assertRaises(ValueError):features([[None]*39]*4)
        with self.assertRaises(ValueError):self.model.score(np.zeros((3,39)))

    def test_report_keeps_episode_denominators(self):
        self.assertEqual(len(self.card['evaluation']['test']['episodes']),4)
        self.assertFalse(self.card['physical_safety_limit'])
        self.assertFalse(self.card['probability_calibrated'])

    def test_hard_reject_and_missing_window_block(self):
        adapter=FakeAdapter()
        monitor=StatisticalMonitoringAgent(adapter.action_space,adapter.observation_space,detector=self.model,
                                           window_provider=lambda:[],enable_experimental=True)
        self.assertEqual(monitor.check_request({'task_id':'T','action':[2]*4},[0]*39)['rule_id'],'R-02')
        self.assertEqual(monitor.check_request({'task_id':'T','action':[0]*4},[0]*39)['rule_id'],'R-03-STAT')
        self.assertEqual(len(monitor.logs),2)
        self.assertFalse(monitor.logs[-1]['approved'])

    def test_explicit_opt_in_required(self):
        adapter=FakeAdapter()
        with self.assertRaises(ValueError):StatisticalMonitoringAgent(adapter.action_space,adapter.observation_space,
                                     detector=self.model,window_provider=lambda:[])


if __name__=='__main__':unittest.main()
