import math
import unittest
from types import SimpleNamespace
import numpy as np
from l3_system.execution.base_adapter import ActionCoordinates
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy


class FakeAdapter:
    def __init__(self):
        self.metadata={'environment_id':'MAbEnvGym','episode_id':'ep','sampling_period':{'value':60}}
        self.state_version=1
        self._xscale=np.ones(1970);self._xscale[17]=3
        self.action_space=SimpleNamespace(shape=(9,),low=np.zeros(9,dtype=np.float32),high=np.ones(9,dtype=np.float32)*50,dtype=np.float32)
        self.observation_space=SimpleNamespace(low=np.zeros(1970),high=np.ones(1970)*50)
        self._env=SimpleNamespace(steady_actions=np.array([1,1,1,1,1,1,1,1,0],dtype=np.float32))
        self.action_coordinates=ActionCoordinates(self.action_space.low,self.action_space.high,np.ones(9),np.zeros(9))
    def validate_action(self,action):return action


class AcknowledgementTests(unittest.TestCase):
    def setUp(self):
        self.a=FakeAdapter()
        self.cfg=dict(setpoint_dm=15.,kp_L_min_per_dm=math.pi*2.5**2/60,
            ki_L_min_per_dm_min=0.,kd_L_min_min_per_dm=0.,sampling_period_min=60.)
        self.kw=dict(startup_level_dm=15.,startup_trusted=True,episode_id='ep',state_version=1,allow_research=True)
        self.p=AcknowledgedBufferPolicy(self.a,self.cfg,**self.kw)
        self.obs=np.ones(1970);self.obs[17]=15.1/3
    def proposal(self):return self.p.predict(self.obs,self.a.action_space)
    def receipt(self,action):
        self.a.state_version+=1
        return dict(success=True,action=action.tolist(),state_version=self.a.state_version,observation=['not used'])
    def test_only_matching_ack_updates_balance(self):
        action=self.proposal();self.p.acknowledge(self.receipt(action));self.p.commit(True)
        self.assertAlmostEqual(self.p.level,15.)
        self.assertEqual(self.p.version,2)
    def test_rejection_does_not_update_or_integrate(self):
        self.proposal();self.p.commit(False)
        self.assertEqual(self.p.version,1);self.assertEqual(self.p.base.integral,0)
    def test_no_receipt_is_not_success(self):
        self.proposal()
        with self.assertRaises(RuntimeError):self.p.commit(True)
    def test_duplicate_receipt_latches(self):
        action=self.proposal();r=self.receipt(action);self.p.acknowledge(r)
        with self.assertRaises(RuntimeError):self.p.acknowledge(r)
        self.assertTrue(self.p.failed)
    def test_changed_action_is_not_assumed_requested(self):
        action=self.proposal();r=self.receipt(action);r['action'][7]+=.01
        with self.assertRaises(RuntimeError):self.p.acknowledge(r)
    def test_lost_chain_after_real_step_latches(self):
        action=self.proposal();self.p.acknowledge(self.receipt(action))
        with self.assertRaises(RuntimeError):self.p.commit(False)
    def test_stale_episode_and_version_block(self):
        self.a.metadata['episode_id']='other'
        with self.assertRaises(RuntimeError):self.proposal()
    def test_missing_measurement_latches(self):
        self.obs[17]=np.nan
        with self.assertRaises(RuntimeError):self.proposal()
        self.assertTrue(self.p.failed)
    def test_large_unmodelled_change_blocks(self):
        self.obs[17]=16./3
        with self.assertRaises(RuntimeError):self.proposal()
    def test_opt_in_and_trusted_startup_required(self):
        for key in ['allow_research','startup_trusted']:
            kw={**self.kw,key:False}
            with self.assertRaises(ValueError):AcknowledgedBufferPolicy(self.a,self.cfg,**kw)
    def test_label_and_true_poststep_not_needed(self):
        action=self.proposal();receipt=self.receipt(action)
        receipt['observation']=None;receipt['reward']='irrelevant'
        self.p.acknowledge(receipt);self.p.commit(True)
        self.assertAlmostEqual(self.p.level,15.)
    def test_new_anchor_needed_after_reset(self):
        self.a.state_version+=1
        with self.assertRaises(RuntimeError):self.p.reset()
