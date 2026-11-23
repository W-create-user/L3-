from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from l3_system.monitoring.sensor_exploration import explore,validate_config,select_threshold
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from check_next_sensor_plan import validate as validate_plan
from check_r03_activation import check

class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/research/sensor_exploration.example.json').read_text());self.config['trees']=10
        rng=np.random.default_rng(15);self.episodes=[]
        for split,n in (('train',2),('validation',3)):
            for i in range(n):
                values=rng.normal(0,.1,(24,39));values[:,0]=np.arange(24);labels=[False]*24
                if split=='validation' and i:
                    values[10:16,6]+=2;labels[10:16]=[True]*6
                eid=f'{split}-{i}'
                self.episodes.append(dict(episode_id=eid,lineage_group=eid,split=split,observations=values.tolist(),
                    labels=labels,label_scope='synthetic_fixture',label_source='synthetic unit fixture'))
    def test_grid_actually_fits_with_validation_fpr_budget(self):
        result=explore(self.episodes,self.config)
        self.assertEqual(len(result['candidates']),8);self.assertFalse(result['test_evaluated'])
        for row in result['candidates']:
            self.assertLessEqual(row['validation']['false_positive_rate'],.05)
            self.assertEqual(row['whole_uncorrupted_validation_episodes'],1)
        self.assertFalse(result['deployment_authorized'])
    def test_test_split_explicitly_rejected(self):
        item=deepcopy(self.episodes[-1]);item.update(split='test',episode_id='hidden',lineage_group='hidden')
        with self.assertRaisesRegex(ValueError,'test data forbidden'):explore(self.episodes+[item],self.config)
    def test_clock_feature_and_bad_features_rejected(self):
        for kind,value in (('projections',{'clock':[0]}),('feature_sets',{'invalid':['label']})):
            config=deepcopy(self.config);config[kind]=value
            with self.assertRaises(ValueError):validate_config(config)
    def test_changed_clock_cannot_disguise_duplicate_trajectory(self):
        item=deepcopy(self.episodes[0]);item.update(split='validation',episode_id='copy',lineage_group='copy')
        for row in item['observations']:row[0]+=100
        with self.assertRaisesRegex(ValueError,'identical non-clock'):explore(self.episodes+[item],self.config)
    def test_training_fault_and_absent_whole_clean_controls_rejected(self):
        rows=deepcopy(self.episodes);rows[0]['labels'][1]=True
        with self.assertRaises(ValueError):explore(rows,self.config)
        with self.assertRaises(ValueError):explore([e for e in self.episodes if e['episode_id']!='validation-0'],self.config)
    def test_threshold_zero_fpr_can_abstain_all(self):
        threshold,metrics=select_threshold([False,True],[.5,.5],0)
        self.assertGreater(threshold,.5);self.assertEqual(metrics['false_positive_rate'],0)
        self.assertEqual(metrics['recall'],0)

class NextPlanTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads((ROOT/'configs/research/sensor_benchmark_next.draft.json').read_text())
    def test_nonexecutable_plan(self):
        result=validate_plan(self.plan);self.assertEqual(result['planned_steps'],1920)
        self.assertFalse(result['data_collected']);self.assertFalse(result['execution_authorized'])
    def test_training_contamination_and_enabled_plan_rejected(self):
        self.plan['episodes'][0]['fault']=deepcopy(self.plan['episodes'][-1]['fault'])
        with self.assertRaises(ValueError):validate_plan(self.plan)
        self.plan['episodes'][0]['fault']=None
        self.plan['execution_authorized']=True
        with self.assertRaises(ValueError):validate_plan(self.plan)
    def test_activation_template_never_authorizes(self):
        review=json.loads((ROOT/'configs/research/r03_activation_review.draft.json').read_text())
        result=check(review);self.assertTrue(result['missing_fields'])
        for field in result['missing_fields']:review[field]=.1 if field=='relative_upper' else 'declared-reference'
        result=check(review);self.assertTrue(result['declarations_complete'])
        self.assertFalse(result['approval_verified']);self.assertFalse(result['export_allowed'])
        review['runtime_enabled']=True
        with self.assertRaises(ValueError):check(review)

if __name__=='__main__':unittest.main()
