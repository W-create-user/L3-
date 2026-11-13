from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
import test_dataset_intake as intake_fixtures
from train_from_intake import prepare
from l3_system.monitoring.binding import make_binding,check_window
from l3_system.monitoring.anomaly_detector import AnomalyDetector
from l3_system.monitoring.statistical_monitor import StatisticalMonitoringAgent
from l3_system.tests.test_live_orchestration_contracts import FakeBox


class TrainingBridgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=intake_fixtures.DatasetIntakeTests('test_valid_structure');self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.labels={'schema_version':'D03Labels/1.0','scope':'sensor_fault','alignment':'observation_before',
                     'split_reviewed_by':'synthetic-review-fixture','episodes':{
                         e['episode_id']:{'labels':[False,False],'source_ref':'synthetic independent test labels',
                                          'reviewed_by':'synthetic-review-fixture','normal_basis':'synthetic fixture baseline'}
                         for e in f.manifest['episodes']}}

    def prepare(self):
        f=self.fixture;return prepare(f.manifest,f.root,f.config_path,f.dictionary_path,self.labels)

    def test_bridge_uses_pre_action_observation_only(self):
        episodes,binding,receipt=self.prepare()
        self.assertEqual(episodes[0]['observations'],[[0,0],[0,0]])
        self.assertNotIn('reward',episodes[0]);self.assertNotIn('failure_reason',episodes[0])
        self.assertEqual(binding['environment'],'Atropine');self.assertTrue(receipt['split_reviewed_by'])

    def test_null_or_interface_label_scope_refused(self):
        self.labels['episodes']['fixture-0']['labels']=[None,None]
        with self.assertRaises(ValueError):self.prepare()
        self.labels['scope']='interface_execution'
        with self.assertRaises(ValueError):self.prepare()

    def test_review_and_normal_basis_required(self):
        self.labels['episodes']['fixture-0']['normal_basis']=''
        with self.assertRaises(ValueError):self.prepare()

    def test_exact_label_coverage_required(self):
        del self.labels['episodes']['fixture-1']
        with self.assertRaises(ValueError):self.prepare()

    def test_label_alignment_cannot_use_future_result(self):
        self.labels['alignment']='observation_after'
        with self.assertRaises(ValueError):self.prepare()

    def test_label_summary_conflict_rejected(self):
        self.labels['episodes']['fixture-0']['labels'][0]=True
        with self.assertRaises(ValueError):self.prepare()

    def test_binding_rejects_same_dimension_different_semantics(self):
        _,binding,_=self.prepare()
        context={'binding':binding,'episode_id':'E','state_version':3}
        packet={'binding':deepcopy(binding),'episode_id':'E','state_versions':[0,1,2,3],'values':[[0,0]]*4}
        self.assertEqual(check_window(packet,context,binding,4),[[0,0]]*4)
        for key,value in [('unit','wrong'),('coordinate','wrong'),('name','swapped')]:
            bad=deepcopy(packet);bad['binding']['observation'][0][key]=value
            with self.assertRaises(ValueError):check_window(bad,context,binding,4)

    def test_context_episode_and_versions_required(self):
        _,binding,_=self.prepare();context={'binding':binding,'episode_id':'E','state_version':3}
        packet={'binding':binding,'episode_id':'E','state_versions':[0,1,2,3],'values':[[0,0]]*4}
        for versions in ([0,1,1,3],[1,2,3,4],[0,1,2,True]):
            with self.assertRaises(ValueError):check_window(dict(packet,state_versions=versions),context,binding,4)
        with self.assertRaises(ValueError):check_window(dict(packet,episode_id='old'),context,binding,4)

    def test_whole_intake_to_bound_training(self):
        f=self.fixture
        for number,episode in enumerate(f.manifest['episodes']):
            eid=episode['episode_id'];rows=[]
            for i in range(4):
                row=deepcopy(f.rows[eid][0]);row.update(step_index=i,simulation_time=i,
                    collected_at=f'2026-11-12T06:00:0{i}Z',state_version_before=i+1,state_version_after=i+2,
                    observation_before=[number*0.01+i*0.001]*2)
                rows.append(row)
            f.rows[eid]=rows;episode['step_count']=4
            truth=[False]*4
            if episode['split']!='train':
                truth[-1]=True;episode.update(label_summary='injected_anomaly',event_start_step=3,event_end_step=3)
            self.labels['episodes'][eid]['labels']=truth
        f.sync();episodes,binding,receipt=self.prepare()
        model=AnomalyDetector(window_size=2);model.fit(episodes,label_scope='sensor_fault',binding=binding,receipt=receipt)
        self.assertEqual(model.card['binding'],binding)
        self.assertEqual(model.card['threshold_source'],'validation_only')
        saved=f.root/'bound_model';model.save(saved)
        self.assertEqual(AnomalyDetector.load_trusted(saved,expected_binding=binding).binding,binding)
        wrong=deepcopy(binding);wrong['observation'][0]['unit']='wrong'
        with self.assertRaises(ValueError):AnomalyDetector.load_trusted(saved,expected_binding=wrong)
        context={'binding':binding,'episode_id':'live-fixture','state_version':3}
        packet={'binding':binding,'episode_id':'live-fixture','state_versions':[2,3],'values':[[0,0],[0,0]]}
        monitor=StatisticalMonitoringAgent(FakeBox([-1],[1]),SimpleNamespace(shape=(2,)),detector=model,
            window_provider=lambda:packet,trusted_context_provider=lambda:context,enable_experimental=True)
        with patch.object(model,'score',return_value={'anomalous':False,'anomaly_score':0.0,'threshold':1.0}):
            self.assertEqual(monitor.check_request({'task_id':'T','action':[0],'state_version':3},[0,0])['decision'],'pass')
            self.assertEqual(monitor.check_request({'task_id':'T','action':[0],'state_version':2},[0,0])['decision'],'reject')
            packet['episode_id']='old-episode'
            self.assertEqual(monitor.check_request({'task_id':'T','action':[0],'state_version':3},[0,0])['decision'],'reject')

    def test_int_float_duplicate_trajectories_rejected(self):
        episodes,binding,receipt=self.prepare()
        episodes[0]['observations']=[[0,0],[0,0]]
        episodes[1]['observations']=[[0.0,0.0],[0.0,0.0]]
        with self.assertRaisesRegex(ValueError,'identical trajectory'):
            AnomalyDetector(window_size=2).fit(episodes,label_scope='sensor_fault')
