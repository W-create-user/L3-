import unittest
from experiments.audit_b08_development import paired_estimate

class PairedStatisticsTests(unittest.TestCase):
    def test_ratio_of_means_not_mean_of_ratios(self):
        r=paired_estimate([1,10],[1,5],['same','same'],resamples=100)
        self.assertAlmostEqual(r['ratio_of_paired_means_reduction'],1-6/11)
        self.assertNotAlmostEqual(r['ratio_of_paired_means_reduction'],.25)
        self.assertFalse(r['formal_target_acceptance'])
    def test_nonpaired_and_nonfinite_cycles_fail(self):
        with self.assertRaises(ValueError):paired_estimate([1,2],[1],['x','y'])
        with self.assertRaises(ValueError):paired_estimate([1],[float('nan')],['x'])
    def test_fixed_strata_keep_deterministic_conditions(self):
        r=paired_estimate([180,360],[180,180],['normal','bias'],resamples=100)
        self.assertAlmostEqual(r['ratio_of_paired_means_reduction'],1/3)
        self.assertAlmostEqual(r['stratified_paired_bootstrap_95_interval'][0],1/3)
        self.assertFalse(r['independence_verified'])
