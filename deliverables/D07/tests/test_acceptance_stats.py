import unittest
from l3_system.monitoring.mab_acceptance_stats import analyze
class StatsTests(unittest.TestCase):
 def data(self):
  p=[];r=[]
  for i,(b,c) in enumerate([(60,30),(180,180)]):
   for arm,t in [('baseline',b),('candidate',c)]:
    spec=dict(run_id=f'{i}-{arm}',pair_id=str(i),condition='normal',group_id=str(i),arm=arm);p.append(spec);r.append({**spec,'started':True,'success':True,'simulated_makespan_min':t,'wall_seconds':1.})
  return p,r
 def test_ratio_of_means_not_average_of_ratios(self):
  p,r=self.data();out=analyze(p,r,resamples=100);self.assertAlmostEqual(out['efficiency']['common_success']['point']['improvement'],.125)
 def test_failure_and_missing_receipt_stay_in_plan(self):
  p,r=self.data();r[1].update(success=False,simulated_makespan_min=None,failure_reason='timeout');r.pop()
  o=analyze(p,r,resamples=100);self.assertEqual(o['arms']['candidate']['planned'],2);self.assertEqual(o['arms']['candidate']['successes'],0);self.assertEqual(o['arms']['candidate']['unknown_start'],1);self.assertEqual(len(o['failure_details']),2)
 def test_wall_timeout_or_missing_time_cannot_claim_success(self):
  p,r=self.data();r[0]['wall_seconds']=90.01
  with self.assertRaises(ValueError):analyze(p,r,resamples=100)
  r[0].pop('wall_seconds')
  with self.assertRaises(ValueError):analyze(p,r,resamples=100)
 def test_success_after_deadline_is_rejected(self):
  p,r=self.data();r[0]['simulated_makespan_min']=601
  with self.assertRaises(ValueError):analyze(p,r,resamples=100)
if __name__=='__main__':unittest.main()
