import copy
import json
from pathlib import Path
import unittest
import numpy as np
from data_generation.transforms import sensor_view,feature_payload
from data_generation.validation import validate_config

class B05TransformTests(unittest.TestCase):
    def test_injection_does_not_mutate_true_state_and_has_independent_rng(self):
        original=[10.,20.];fault={'kind':'bias','first_step':2,'last_step':3,'multiplier':1.2}
        view,mask,label,event=sensor_view(original,index=1,step=2,fault=fault,rng=np.random.default_rng(1))
        self.assertEqual(original,[10.,20.]);self.assertEqual(view,[10.,24.]);self.assertEqual(label['sensor_fault_label'],1);self.assertIsNone(label['physical_health_label'])
        quiet=sensor_view(original,index=1,step=1,fault=fault,rng=np.random.default_rng(1));self.assertEqual(quiet[2]['sensor_fault_label'],0)
    def test_missing_is_null_with_explicit_mask_not_zero(self):
        fault={'kind':'missing','first_step':1,'last_step':2}
        view,mask,label,event=sensor_view([12.],index=0,step=1,fault=fault,rng=np.random.default_rng(2))
        self.assertEqual(view,[None]);self.assertEqual(mask,[True]);self.assertEqual(feature_payload(view,mask,indices=[0],scale=1,unit='dimensionless')['text'],'measurement unavailable')
    def test_noise_replays_from_seed(self):
        fault={'kind':'noise','first_step':1,'last_step':2,'sigma':.2}
        a=sensor_view([12.],index=0,step=1,fault=fault,rng=np.random.default_rng(4))
        b=sensor_view([12.],index=0,step=1,fault=fault,rng=np.random.default_rng(4))
        self.assertEqual(a,b)
    def test_features_have_no_label_episode_seed_or_split_fields(self):
        features=feature_payload([13.,14.],[False,False],indices=[1],scale=1,unit='dimensionless')
        self.assertEqual(features['series'],[14.]);self.assertNotIn('fault',features['text']);self.assertNotIn('episode_id',features)
    @unittest.skipUnless((Path(__file__).resolve().parents[2]/'configs/scenarios/b05_frozen_20261120.json').is_file(), 'Private B05 protocol is not included in code-only publication')
    def test_bad_config_cannot_contaminate_training_or_duplicate_lineage_seed(self):
        config=json.loads((Path(__file__).resolve().parents[2]/'configs/scenarios/b05_frozen_20261120.json').read_text())
        self.assertTrue(validate_config(config))
        for mode in ['fault_train','duplicate']:
            bad=copy.deepcopy(config)
            if mode=='fault_train':bad['episodes'][0]['fault']={'kind':'missing','first_step':1,'last_step':2}
            else:bad['episodes'].append(copy.deepcopy(bad['episodes'][0]))
            with self.assertRaises(ValueError):validate_config(bad)

    @unittest.skipUnless((Path(__file__).resolve().parents[2]/'configs/scenarios/b05_frozen_20261120.json').is_file(), 'Private B05 protocol is not included in code-only publication')
    def test_primary_mab_fault_coverage_cannot_be_replaced_by_auxiliary_data(self):
        config=json.loads((Path(__file__).resolve().parents[2]/'configs/scenarios/b05_frozen_20261120.json').read_text())
        config['episodes']=[row for row in config['episodes'] if row['environment_id']!='MAbEnvGym' or row['fault'] is None]
        with self.assertRaises(ValueError):validate_config(config)
