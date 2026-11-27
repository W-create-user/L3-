import copy
from datetime import datetime,timezone
import unittest
import numpy as np
from l3_system.tests.test_acknowledged_buffer import FakeAdapter
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy
from l3_system.execution.controllers.receipt_boundary import ReceiptBoundary

class ExecutableFake(FakeAdapter):
    def __init__(self):
        super().__init__();self.step_count=0;self.step_calls=0
    def step(self,action):
        self.step_count+=1;self.step_calls+=1;self.state_version+=1
        return dict(success=True,action=np.asarray(action).tolist(),state_version=self.state_version,
                    step_count=self.step_count,observation=['EVALUATOR_ONLY'],reward=-100,fault_labels=['PRIVATE'])


def fixture():
    adapter=ExecutableFake();clock=[0.]
    cfg=dict(setpoint_dm=15.,kp_L_min_per_dm=np.pi*2.5**2/60,ki_L_min_per_dm_min=0.,kd_L_min_min_per_dm=0.,sampling_period_min=60.)
    policy=AcknowledgedBufferPolicy(adapter,cfg,startup_level_dm=15.,startup_trusted=True,episode_id='ep',state_version=1,allow_research=True)
    cert=dict(episode_id='ep',state_version=1,timestamp=datetime.now(timezone.utc).isoformat(),level_dm=15.,calibration_source='validated_simulator_reset_before_fault_injection')
    boundary=ReceiptBoundary(adapter,policy,cert,receipt_timeout_seconds=.5,clock=lambda:clock[0])
    obs=np.ones(1970);obs[17]=15.1/3
    action=boundary.predict(obs,adapter.action_space)
    req=dict(request_id='request-1',episode_id='ep',state_version=1,step_count=0,action=action.tolist())
    boundary.bind_request(req)
    return adapter,policy,boundary,action,clock,req


class BoundaryTests(unittest.TestCase):
    def assert_stops_extra_step(self,b,a,action):
        before=a.step_calls
        with self.assertRaises(RuntimeError):b.step(action)
        self.assertEqual(a.step_calls,before);self.assertTrue(b.failed)
        with self.assertRaises(RuntimeError):b.predict(np.ones(1970),a.action_space)
    def test_correct_receipt_is_allowlisted_and_committed(self):
        a,p,b,action,clock,req=fixture();b.bind_request(req)
        result=b.step(action);receipt=b.make_receipt(result)
        self.assertEqual(set(receipt),b.RECEIPT_FIELDS)
        seen=[];ack=p.acknowledge
        def spy(data):seen.append(copy.deepcopy(data));ack(data)
        p.acknowledge=spy;b.acknowledge(receipt);b.commit(True)
        self.assertEqual(set(seen[0]),{'success','action','state_version'})
        self.assertEqual(a.step_calls,1);self.assertFalse(b.failed)
    def test_malformed_and_mismatched_receipts_latch(self):
        mutations={
            'none':lambda r:None,'list':lambda r:[],
            'missing_action':lambda r:{k:v for k,v in r.items() if k!='action'},
            'numeric_string':lambda r:{**r,'action':[str(x) for x in r['action']]},
            'malformed_action':lambda r:{**r,'action':['broken']*9},
            'wrong_shape':lambda r:{**r,'action':[0.]},
            'nan':lambda r:{**r,'action':[float('nan')]*9},
            'wrong_action':lambda r:{**r,'action':[0.]*9},
            'wrong_episode':lambda r:{**r,'episode_id':'other'},
            'wrong_request':lambda r:{**r,'request_id':'other'},
            'wrong_version':lambda r:{**r,'state_version':99},
            'wrong_step':lambda r:{**r,'step_count':99},
            'float_version':lambda r:{**r,'state_version':float(r['state_version'])},
            'not_executed':lambda r:{**r,'success':False},
            'integer_success':lambda r:{**r,'success':1},
            'truth_in_receipt':lambda r:{**r,'observation':[1.]},
        }
        for name,change in mutations.items():
            with self.subTest(name=name):
                a,p,b,action,clock,req=fixture();receipt=b.make_receipt(b.step(action))
                with self.assertRaises(RuntimeError):b.acknowledge(change(receipt))
                self.assertTrue(p.failed);self.assert_stops_extra_step(b,a,action)
    def test_missing_receipt_blocks_next_native_step(self):
        a,p,b,action,clock,req=fixture();b.step(action)
        self.assert_stops_extra_step(b,a,action)
    def test_missing_acknowledgement_blocks_commit(self):
        a,p,b,action,clock,req=fixture();b.step(action)
        with self.assertRaises(RuntimeError):b.commit(True)
        self.assert_stops_extra_step(b,a,action)
    def test_expired_receipt_latches(self):
        a,p,b,action,clock,req=fixture();r=b.make_receipt(b.step(action));clock[0]=1
        with self.assertRaises(RuntimeError):b.acknowledge(r)
        self.assert_stops_extra_step(b,a,action)
    def test_duplicate_receipt_cannot_repeat_step(self):
        a,p,b,action,clock,req=fixture();r=b.make_receipt(b.step(action));b.acknowledge(r)
        with self.assertRaises(RuntimeError):b.acknowledge(r)
        self.assert_stops_extra_step(b,a,action)
    def test_context_reset_makes_receipt_stale(self):
        a,p,b,action,clock,req=fixture();r=b.make_receipt(b.step(action));a.metadata['episode_id']='new'
        with self.assertRaises(RuntimeError):b.acknowledge(r)
        self.assert_stops_extra_step(b,a,action)
    def test_unreviewed_or_changed_request_never_steps(self):
        a,p,b,action,clock,req=fixture();req['request_id']='changed'
        with self.assertRaises(RuntimeError):b.bind_request(req)
        self.assert_stops_extra_step(b,a,action);self.assertEqual(a.step_calls,0)
    def test_rejection_before_native_call_does_not_advance(self):
        a,p,b,action,clock,req=fixture();b.commit(False)
        self.assertEqual(a.step_calls,0);self.assertEqual(p.version,1);self.assertFalse(b.failed)
    def test_no_reset_clears_unknown_result(self):
        a,p,b,action,clock,req=fixture();b.step(action)
        with self.assertRaises(RuntimeError):b.commit(False)
        with self.assertRaises(RuntimeError):b.reset()
        self.assert_stops_extra_step(b,a,action)
    def test_native_exception_latches_even_without_receipt(self):
        a,p,b,action,clock,req=fixture()
        original=a.step
        def lose(action):
            original(action)
            raise RuntimeError('lost after native step')
        a.step=lose
        with self.assertRaises(RuntimeError):b.step(action)
        self.assert_stops_extra_step(b,a,action)
        self.assertEqual(a.step_calls,1)
    def test_certificate_must_match_the_reset(self):
        a,p,b,action,clock,req=fixture()
        for key,value in [('episode_id','other'),('state_version',999),('level_dm',14.),('timestamp','2026-11-26T00:00:00'),('calibration_source','unknown')]:
            certificate={**b.certificate,key:value}
            with self.assertRaises(ValueError):ReceiptBoundary(a,p,certificate,receipt_timeout_seconds=.5)
