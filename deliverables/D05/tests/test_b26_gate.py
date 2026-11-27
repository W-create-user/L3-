import time,unittest
from types import SimpleNamespace
import numpy as np
from l3_system.monitoring.b26_learning_gate import B26LearningGate,FailClosedPolicy,validate_anchor
class B26GateTests(unittest.TestCase):
 def gate(self):
  g=B26LearningGate.__new__(B26LearningGate);g.config={'model_sha256':'fixture'};g.cache={};g.values=[[0]]*4;g.versions=[1,2,3,4];g.episode='e';g.backend=SimpleNamespace(state_version=4);g.last={'risk_level':1};g.failed=False;g.events=[];g._current=lambda:True;g.count=0
  g.hard=SimpleNamespace(check_request=lambda r,o:{'decision':'pass','task_id':r['task_id'],'request_id':r['request_id']})
  def score(ident):g.count+=1;return {'status':'scored','anomalous':False}
  g._score=score;g._fault=lambda reason:setattr(g,'failed',True)
  return g
 def request(self):return {'request_id':'r','task_id':'t','subtask_id':'s','episode_id':'e','state_version':4,'action':[0]*9}
 def test_repeat_review_is_idempotent_without_extra_model_call(self):
  g=self.gate();r=self.request();first=g.check_request(r);second=g.check_request(r);self.assertEqual(first,second);self.assertEqual(g.count,1)
 def test_changed_reused_request_cannot_use_cached_pass(self):
  g=self.gate();r=self.request();g.check_request(r);r['action'][0]=1
  self.assertEqual(g.check_request(r)['decision'],'reject');self.assertTrue(g.failed)
 def test_stale_state_cannot_use_cached_pass(self):
  g=self.gate();r=self.request();g.check_request(r);g._current=lambda:False
  self.assertEqual(g.check_request(r)['decision'],'reject')
 def test_hard_rejection_cannot_be_overridden_by_cache(self):
  g=self.gate();r=self.request();g.check_request(r);g.hard.check_request=lambda r,o:{'decision':'reject'}
  self.assertEqual(g.check_request(r)['decision'],'reject');self.assertEqual(g.count,1)
 def test_every_candidate_exception_latches_outer_guard(self):
  class Base:
   def acknowledge(self,r):raise ValueError('bad typed receipt')
   def predict(self,*a):return 'should not reach'
  p=FailClosedPolicy(Base())
  with self.assertRaises(ValueError):p.acknowledge({})
  with self.assertRaises(RuntimeError):p.predict([],None)
  self.assertTrue(p.failed)
 def test_anchor_is_current_prefault_and_has_source(self):
  obs=np.zeros(1970);obs[17]=5;a=SimpleNamespace(metadata={'episode_id':'e'},state_version=1,get_observation=lambda:obs,_xscale=np.ones(1970)*3)
  c={'episode_id':'e','state_version':1,'unit':'dm','coordinate':'physical_projection','kind':'trusted_simulation_reset_before_fault','source_ref':'fixture','fault_started':False,'observed_monotonic':time.monotonic(),'level_dm':15.}
  self.assertEqual(validate_anchor(c,a),15)
  for change in [{'fault_started':True},{'source_ref':''},{'level_dm':15.1},{'state_version':2},{'observed_monotonic':time.monotonic()-6}]:
   with self.assertRaises(ValueError):validate_anchor({**c,**change},a)
if __name__=='__main__':unittest.main()
