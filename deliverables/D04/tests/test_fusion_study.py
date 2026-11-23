import unittest
from unittest.mock import patch
from l3_system.monitoring.fusion_study import study
from l3_system.monitoring.fusion_comparison import VARIANTS
from l3_system.monitoring.multimodal_model import synthetic_batches

class StudyTests(unittest.TestCase):
    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError):study(synthetic_batches(),seeds=(17,17),scope='synthetic_engineering_only')
    def test_unknown_scope_rejected(self):
        with self.assertRaises(ValueError):study(synthetic_batches(),scope='physical_safety')
    def test_fixed_common_cohort_and_distinct_seeds(self):
        fake={'variants':{v:{'test':{'f1':.5}} for v in VARIANTS}}
        with patch('l3_system.monitoring.fusion_study.compare',return_value=fake) as call:
            report=study(synthetic_batches(),scope='synthetic_engineering_only')
        self.assertEqual([c.kwargs['seed'] for c in call.call_args_list],[17,29,43])
        self.assertTrue(all(c.kwargs['common_test_cohort'] for c in call.call_args_list))
        self.assertEqual(report['trained_models'],21)
        self.assertEqual(report['summary']['full_concat']['test_f1_initialization_sd'],0)
        self.assertFalse(report['physical_efficacy_established'])
