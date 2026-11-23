import unittest
from l3_system.monitoring.paired_analysis import analyze

class PairedAnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(pair_id=str(i),group_id='g'+str(i//2),condition='nominal',arm=arm,
                     success=arm=='candidate',simulated_makespan=10 if arm=='baseline' else 8,evidence_ref='fixture')
                for i in range(4) for arm in ('baseline','candidate')]
    def test_failures_count_and_groups_are_units(self):
        result=analyze(self.rows(),resamples=100)
        self.assertEqual(result['paired_episode_count'],4);self.assertEqual(result['lineage_group_count'],2)
        self.assertEqual(result['success_rate_difference'],1)
        self.assertEqual(result['success_difference_percentile_95_interval'],[1,1])
        self.assertIsNone(result['mean_relative_makespan_reduction_complete_success_pairs'])
        self.assertEqual(result['time_pair_exclusion_count'],4)
        self.assertTrue(result['small_cluster_warning'])
    def test_time_reduction_not_success_claim(self):
        rows=self.rows()
        for row in rows:row['success']=True
        result=analyze(rows,resamples=100)
        self.assertAlmostEqual(result['mean_relative_makespan_reduction_complete_success_pairs'],.2)
        self.assertFalse(result['target_achievement_claimed'])
    def test_unpaired_and_duplicate_rejected(self):
        for rows in (self.rows()[:-1],self.rows()+[self.rows()[0]]):
            with self.assertRaises(ValueError):analyze(rows,resamples=100)
    def test_pair_mismatch_rejected(self):
        rows=self.rows();rows[1]['condition']='fault'
        with self.assertRaises(ValueError):analyze(rows,resamples=100)
    def test_unknown_outcome_not_treated_as_success(self):
        rows=self.rows();rows[0]['success']=None
        with self.assertRaises(ValueError):analyze(rows,resamples=100)
    def test_repeated_calls_reproducible(self):
        rows=self.rows();rows[1]['success']=False
        self.assertEqual(analyze(rows,resamples=100),analyze(rows,resamples=100))
    def test_input_order_does_not_change_sampling(self):
        rows=self.rows();rows[1]['success']=False
        self.assertEqual(analyze(rows,resamples=100),analyze(list(reversed(rows)),resamples=100))

    def test_invalid_time_rejected(self):
        for value in (True,0,float('nan')):
            rows=self.rows();rows[0]['simulated_makespan']=value
            with self.assertRaises(ValueError):analyze(rows,resamples=100)

if __name__=='__main__':unittest.main()
