"""Fake-state protocol checks; not SMPL trajectories or physical efficacy."""
from pathlib import Path
import tempfile
import unittest
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.monitoring.atropine_experiment import AtropineExperimentSession,ObservedAdapter

class AtropineExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.clock=[1.];self.adapter=FakeAdapter()
        self.session=AtropineExperimentSession(self.adapter,Path(self.temp.name)/'session',enable_approved_experiment=True,
            fixture=True,clock=lambda:self.clock[0])
        self.session.start();self.adapter._obs[5]=10.;self.adapter._obs[6]=10.;self.session.source._capture()
    def breach(self):
        self.adapter._obs[6]=12.;self.session.source._capture()
        self.session.run_cycle();self.assertTrue(self.session.latched)
        self.assertEqual(self.adapter.step_calls,0)
    def fake_new_sample(self,current=10.):
        # A test-only trusted source revision; no hidden real-simulator step.
        self.adapter.state_version+=1;self.adapter.step_count+=1
        self.adapter._obs[6]=current;self.session.source._capture()
    def test_versioned_config_matches_experiment_code(self):
        import json
        from l3_system.monitoring.atropine_experiment import experiment_config
        root=Path(__file__).resolve().parents[3]
        self.assertEqual(json.loads((root/'configs/safety/atropine_experiment_20261116.json').read_text()),experiment_config())
    def test_optin_and_real_backend_guard(self):
        with self.assertRaises(ValueError):AtropineExperimentSession(self.adapter,Path(self.temp.name)/'off')
        with self.assertRaises(ValueError):ObservedAdapter(self.adapter)
    def test_equality_allows_single_reviewed_step(self):
        self.adapter._obs[6]=11.;self.session.source._capture()
        self.session.run_cycle();self.assertEqual(self.adapter.step_calls,1)
        # FakeAdapter increments all components; post-step remains within the ratio.
        self.assertFalse(self.session.latched)
    def test_preflight_breach_blocks_and_reset_not_recovery(self):
        self.breach()
        with self.assertRaises(RuntimeError):self.session.run_cycle()
        with self.assertRaises(RuntimeError):self.session.start()
        with self.assertRaises(RuntimeError):self.session.source.step([0.]*4)
        self.assertEqual(self.adapter.step_calls,0)
    def test_three_states_resources_and_human_then_fresh_action_gate(self):
        self.breach()
        for i in range(3):
            self.fake_new_sample();self.assertEqual(self.session.observe_recovery()['fresh_states'],i+1)
            self.assertFalse(self.session.observe_recovery()['new_sample'])
        self.session.system.scheduler.active_agent_tasks['fixture']='pending'
        with self.assertRaises(RuntimeError):self.session.resume(operator='fixture',reason='still locked',cause_reviewed=True)
        self.session.system.scheduler.active_agent_tasks.clear()
        with self.assertRaises(RuntimeError):self.session.resume(operator='fixture',reason='no cause review')
        result=self.session.resume(operator='fixture',reason='cause reviewed; source restored',cause_reviewed=True)
        self.assertFalse(result['executed']);self.assertEqual(self.adapter.step_calls,0)
        self.adapter._obs[6]=12.;self.session.source._capture();self.session.run_cycle()
        self.assertTrue(self.session.latched);self.assertEqual(self.adapter.step_calls,0)
    def test_duplicate_and_one_bad_state_cannot_fill_recovery(self):
        self.breach();self.fake_new_sample();self.session.observe_recovery()
        for _ in range(5):self.assertEqual(self.session.observe_recovery()['fresh_states'],1)
        with self.assertRaises(RuntimeError):self.session.resume(operator='fixture',reason='one state only',cause_reviewed=True)
        self.fake_new_sample(current=12.)
        with self.assertRaises(RuntimeError):self.session.observe_recovery()
        self.assertEqual(self.session.recovery,[])
    def test_stale_missing_or_unobserved_source_cannot_pass(self):
        self.clock[0]=4.;self.session.run_cycle();self.assertTrue(self.session.latched)
        self.assertEqual(self.adapter.step_calls,0)
    def test_external_untracked_version_or_config_change_blocks(self):
        self.adapter.state_version+=1;self.session.run_cycle()
        self.assertTrue(self.session.latched);self.assertEqual(self.adapter.step_calls,0)
        other=FakeAdapter()
        changed=AtropineExperimentSession(other,Path(self.temp.name)/'changed-config',enable_approved_experiment=True,fixture=True,clock=lambda:1.)
        changed.start();other._obs[5]=10.;other._obs[6]=10.;changed.source._capture()
        other.action_space.low[0]=-2.
        changed.run_cycle();self.assertTrue(changed.latched);self.assertEqual(other.step_calls,0)
    def test_post_step_breach_latches_without_undo_or_second_step(self):
        original=self.adapter.step
        def injected(action):
            result=original(action);self.adapter._obs[6]=12.;result['observation']=self.adapter.get_observation();return result
        self.adapter.step=injected
        self.session.run_cycle();self.assertTrue(self.session.latched);self.assertEqual(self.adapter.step_calls,1)
        with self.assertRaises(RuntimeError):self.session.run_cycle()
        self.assertEqual(self.adapter.step_calls,1)
    def test_invalid_reference_and_censoring_do_not_pass(self):
        for ref in (0.,20.):
            self.adapter._obs[5]=ref;self.session.source._capture()
            result=self.session.monitor.check_observation(self.adapter.get_observation())
            self.assertFalse(result['approved'])
    def test_unknown_execution_not_retried(self):
        def broken(action):
            self.adapter.step_calls+=1;self.adapter._ready=False
            raise RuntimeError('fixture uncertain execution')
        self.adapter.step=broken;self.session.run_cycle()
        self.assertTrue(self.session.latched);self.assertTrue(self.session.source.unknown_result)
        self.assertEqual(self.adapter.step_calls,1)
        with self.assertRaises(RuntimeError):self.session.run_cycle()
    def test_controller_change_cannot_bypass_approved_baseline(self):
        with self.assertRaises(RuntimeError):self.session.source.step([.1]*4)
        self.assertEqual(self.adapter.step_calls,0)
    def test_rule_context_before_raw_step_cannot_be_bypassed(self):
        self.adapter._obs[6]=12.;self.session.source._capture()
        with self.assertRaises(RuntimeError):self.session.source.step([0.]*4)
        self.assertEqual(self.adapter.step_calls,0);self.assertTrue(self.session.latched)
    def test_reference_data_age_invalidates_recovery(self):
        self.breach()
        for _ in range(3):self.fake_new_sample();self.session.observe_recovery()
        self.clock[0]=4.
        with self.assertRaises(RuntimeError):self.session.resume(operator='fixture',reason='stale review',cause_reviewed=True)
        self.assertEqual(self.session.recovery,[])
    def test_pending_request_prevents_human_resume(self):
        self.breach()
        for _ in range(3):self.fake_new_sample();self.session.observe_recovery()
        self.session.executor._pending['fixture']={}
        with self.assertRaises(RuntimeError):self.session.resume(operator='fixture',reason='pending not cleared',cause_reviewed=True)
        self.assertTrue(self.session.latched)

if __name__=='__main__':unittest.main()
