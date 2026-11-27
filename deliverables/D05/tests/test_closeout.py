import json,tempfile,time,unittest
from pathlib import Path
import numpy as np
from l3_system.monitoring.b05_model import B05Model
from l3_system.monitoring.buffer_observer import BufferObserver
from l3_system.monitoring.inference_service import InferenceService

class CloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'model.json';rng=np.random.default_rng(8)
        def episode(eid,split,positive=False):
            ws=[]
            for _ in range(6):
                w=np.zeros((4,39));w[:,5]=1.;w[:,6]=1.+rng.normal(0,.001,4)+(0.2 if positive else 0)
                ws.append((w.tolist(),positive))
            return dict(episode_id=eid,group_id=eid,split=split,windows=ws)
        cls.model=B05Model('AtropineEnvGym',{'fixture':'explicit-test-binding'}).fit([episode('train','train')],[episode('cal0','validation'),episode('cal1','validation',True)],[episode('thr0','validation'),episode('thr1','validation',True)])
        cls.digest=cls.model.save(cls.path)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def packet(self):
        w=np.zeros((4,39));w[:,5]=1.;w[:,6]=1.
        packet={'binding':self.model.binding,'episode_id':'fixture','state_versions':[1,2,3,4],'values':w.tolist()}
        context={'binding':self.model.binding,'episode_id':'fixture','state_version':4,'observed_monotonic':time.monotonic()}
        return packet,context
    def test_invalid_budget_and_untrusted_model_startup(self):
        for budget in (float('nan'),0,True):
            with self.assertRaises(ValueError):InferenceService(self.path,self.digest,timeout_seconds=budget)
        with self.assertRaises(RuntimeError):InferenceService(self.path,'0'*64)
    def test_missing_window_never_passes(self):
        p,c=self.packet();p['values'][1][6]=None
        r=self.model.score_packet(p,c);self.assertEqual(r['status'],'unavailable');self.assertTrue(r['automatic_execution_blocked'])
    def test_boolean_and_stale_version_block(self):
        p,c=self.packet();p['state_versions'][0]=True
        self.assertEqual(self.model.score_packet(p,c)['status'],'unavailable')
        p,c=self.packet();p['values'][0][6]=True
        self.assertEqual(self.model.score_packet(p,c)['status'],'unavailable')
    def test_binding_and_stale_wall_clock_block(self):
        p,c=self.packet();c['observed_monotonic']-=1
        self.assertEqual(self.model.score_packet(p,c)['status'],'unavailable')
        p,c=self.packet();c['binding']={'wrong':True}
        self.assertEqual(self.model.score_packet(p,c)['status'],'unavailable')
    def test_shape_cannot_map_mab_to_atropine(self):
        p,c=self.packet();p['values']=[[0.]*1970]*4
        self.assertEqual(self.model.score_packet(p,c)['status'],'unavailable')
    def test_real_process_reply_dedup_and_expiry(self):
        with InferenceService(self.path,self.digest) as service:
            p,c=self.packet();r=service.score('first',p,c);self.assertEqual(r['status'],'scored');self.assertFalse(r['execution_authorized'])
            self.assertEqual(service.score('first',p,c)['reason'],'duplicate_or_invalid_request')
            self.assertEqual(service.score('expired',p,c,absolute_deadline=time.monotonic()-1)['reason'],'expired_request')
    def test_actual_timeout_terminates_worker(self):
        with InferenceService(self.path,self.digest,timeout_seconds=.0001) as service:
            p,c=self.packet();r=service.score('too-short',p,c)
            self.assertIn(r['reason'],('inference_timeout','expired_request','late_or_mismatched_result'))
            self.assertTrue(r['automatic_execution_blocked'])
            if r['reason']!='expired_request':self.assertFalse(service.process.is_alive())
    def test_observer_uses_executed_flow_not_measurement_bias(self):
        ob=BufferObserver(startup_level_dm=15.2,startup_trusted=True,episode_id='e')
        self.assertAlmostEqual(ob.propose(15.3,episode_id='e',state_version=0)['estimated_bias_dm'],.1)
        action=np.zeros(9);action[7]=ob.area/60*.2
        ob.commit(executed=True,physical_action=action,state_version=1)
        self.assertAlmostEqual(ob.propose(15.1,episode_id='e',state_version=1)['estimated_level_dm'],15.)
    def test_rejection_does_not_advance_observer(self):
        ob=BufferObserver(startup_level_dm=15,startup_trusted=True,episode_id='e');ob.propose(15.1,episode_id='e',state_version=0)
        ob.commit(executed=False,physical_action=None,state_version=0);self.assertEqual(ob.version,0);self.assertEqual(ob.level,15)
    def test_initial_bias_without_anchor_and_unknown_result_block(self):
        with self.assertRaises(ValueError):BufferObserver(startup_level_dm=15,startup_trusted=False,episode_id='e')
        ob=BufferObserver(startup_level_dm=15,startup_trusted=True,episode_id='e');ob.propose(15.1,episode_id='e',state_version=0)
        with self.assertRaises(ValueError):ob.commit(executed=True,physical_action=[0.]*9,state_version=2)
        with self.assertRaises(ValueError):ob.propose(15,episode_id='e',state_version=0)
if __name__=='__main__':unittest.main()
