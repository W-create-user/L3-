import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from l3_system.execution.base_adapter import ActionCoordinates
from l3_system.execution.execution_agent import ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter, FakeBox

class BackendCoordinateTests(unittest.TestCase):
    def test_physical_offset_and_scaling_have_distinct_meanings(self):
        delta=ActionCoordinates([-1,-2],[1,2],[1,1],[10,20])
        scaled=ActionCoordinates([-1,-2],[1,2],[3,5],[0,0])
        np.testing.assert_allclose(delta.to_physical([.5,1]),[10.5,21])
        np.testing.assert_allclose(scaled.to_physical([.5,1]),[1.5,5])
        for coords in [delta,scaled]:
            for value in [coords.low,coords.high,np.zeros(2)]:
                np.testing.assert_allclose(coords.from_physical(coords.to_physical(value)),value)
                np.testing.assert_allclose(coords.from_normalized(coords.to_normalized(value)),value)
    def test_bad_units_shape_or_normalized_range_fail_explicitly(self):
        with self.assertRaises(ValueError):ActionCoordinates([0],[0],[1],[0])
        with self.assertRaises(ValueError):ActionCoordinates([0],[1],[0],[0])
        coords=ActionCoordinates([0],[1],[2],[0])
        for value in [[2],['x'],[float('nan')],[0,0]]:
            with self.subTest(value=value),self.assertRaises(ValueError):coords.from_normalized(value)
    def test_request_uses_adapter_identity_and_private_metadata_snapshot(self):
        a=FakeAdapter();a.metadata={'environment':'TEST.MAbMetadata','normalize':False,'uss_subtracted':False,'episode_id':'test-episode'}
        with tempfile.TemporaryDirectory() as tmp:
            ex=ExecutionAgent(a,log_path=Path(tmp)/'events.jsonl')
            request=ex.prepare_action({'task_id':'backend-test','task_type':'control_cycle'})
            self.assertEqual(request['environment'],'TEST.MAbMetadata');self.assertFalse(request['uss_subtracted'])
            request['backend_metadata']['episode_id']='changed'
            self.assertEqual(a.metadata['episode_id'],'test-episode')
            self.assertEqual(ex._pending[request['request_id']]['backend_metadata']['episode_id'],'test-episode')

class ControllerStateTests(unittest.TestCase):
    def setUp(self):
        self.space=FakeBox([0]*9,[50]*9);scale=np.ones(1970);scale[17]=3
        self.adapter=SimpleNamespace(metadata={'environment_id':'MAbEnvGym','sampling_period':{'value':60}},
            observation_space=SimpleNamespace(low=np.zeros(1970),high=np.ones(1970)*20),action_space=self.space,
            _xscale=scale,_env=SimpleNamespace(steady_actions=np.ones(9)*20),
            action_coordinates=ActionCoordinates([0]*9,[50]*9,[1]*9,[0]*9),validate_action=lambda a:a)
        self.config={'setpoint_dm':15.,'kp_L_min_per_dm':1.,'ki_L_min_per_dm_min':.1,'kd_L_min_min_per_dm':0.,'sampling_period_min':60}
    def obs(self):
        v=np.zeros(1970);v[17]=16/3;return v
    def test_rejected_candidate_does_not_advance_integral_or_derivative_state(self):
        c=BufferLevelPID(self.adapter,self.config);c.predict(self.obs(),self.space)
        with self.assertRaises(RuntimeError):c.predict(self.obs(),self.space)
        c.commit(False);self.assertEqual(c.integral,0);self.assertIsNone(c.previous_error)
        c.predict(self.obs(),self.space);c.commit(True);self.assertGreater(c.integral,0)
    def test_saturation_has_anti_windup_and_correct_drain_direction(self):
        config=dict(self.config,kp_L_min_per_dm=100.)
        c=BufferLevelPID(self.adapter,config);action=c.predict(self.obs(),self.space)
        self.assertEqual(action[7],50);c.commit(True)
        self.assertEqual(c.integral,0);self.assertEqual(c.saturation_count,1)
    def test_sampling_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):BufferLevelPID(self.adapter,dict(self.config,sampling_period_min=1))

class ReducedMetadataCompatibilityTests(unittest.TestCase):
    def test_explicit_environment_id_is_not_guessed_from_dimensions(self):
        adapter=FakeAdapter();adapter.metadata={'environment_id':'DECLARED_TEST_BACKEND','normalize':False,'uss_subtracted':False}
        with tempfile.TemporaryDirectory() as tmp:
            executor=ExecutionAgent(adapter,log_path=Path(tmp)/'B.jsonl')
            request=executor.prepare_action({'task_id':'explicit-id','task_type':'control_cycle'})
            self.assertEqual(request['environment'],'DECLARED_TEST_BACKEND')
            self.assertFalse(request['uss_subtracted'])
