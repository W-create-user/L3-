import unittest
from experiments.r03_candidate_check import evaluate

class R03CandidateEvidenceTests(unittest.TestCase):
    def sample(self,multiplier):
        obs=[0.]*39;obs[5]=13.;obs[6]=13.*multiplier;return obs
    def evaluate(self,obs,**kw):
        return evaluate(obs,backend=kw.get('backend','AtropineEnvGym'),normalize=kw.get('normalize',False),layout='atropine-default39-v1')
    def test_constructed_pass_and_breach_never_enable_runtime(self):
        for ratio,expected in [(1.05,'within_candidate_band'),(1.11,'candidate_breach')]:
            r=self.evaluate(self.sample(ratio));self.assertEqual(r['status'],expected)
            self.assertFalse(r['runtime_enabled']);self.assertIsNone(r['operational_decision'])
    def test_no_reuse_for_mab_or_normalized_observations(self):
        self.assertEqual(self.evaluate(self.sample(1),backend='MAbEnvGym')['status'],'incompatible_configuration')
        self.assertEqual(self.evaluate(self.sample(1),normalize=True)['status'],'incompatible_configuration')
    def test_unobservable_threshold_and_invalid_data_are_not_passes(self):
        obs=self.sample(1);obs[5]=19
        self.assertEqual(self.evaluate(obs)['status'],'unsupported_reference_or_censoring')
        obs=self.sample(1);obs[6]=float('nan')
        self.assertEqual(self.evaluate(obs)['status'],'invalid_input')
