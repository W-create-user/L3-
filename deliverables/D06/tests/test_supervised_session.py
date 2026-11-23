from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.monitoring.supervised_session import SupervisedSession
from l3_system.monitoring.supervision_console import create_app
from l3_system.monitoring.risk_policy import RiskPolicy,RiskEvidence

class StubModel:
    anomalous=False
    in_scope=True
    def score(self,window):
        return {'anomalous':self.anomalous,'in_empirical_reference_domain':self.in_scope,
                'sensor_bias_probability':.9 if self.anomalous else .1}

class SupervisedSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=FakeAdapter();self.model=StubModel();self.clock=[0.]
        self.s=SupervisedSession(self.base,self.model,Path(self.tmp.name)/'run',fixture=True,clock=lambda:self.clock[0]);self.s.start()
    def warm(self):
        for _ in range(6):self.s.sample(operator='test',reason='manual restricted simulation')
    def confirm(self):self.s.confirm(operator='test',reason='review current state')
    def test_startup_needs_explicit_sampling_and_confirmation(self):
        with self.assertRaises(RuntimeError):self.s.cycle()
        with self.assertRaises((RuntimeError,ValueError)):self.confirm()
        self.assertEqual(self.base.step_calls,0)
        self.warm();self.assertEqual(self.base.step_calls,6)
        with self.assertRaises(RuntimeError):self.s.cycle()
        self.confirm();self.assertEqual(self.base.step_calls,6)
        self.assertEqual(self.s.cycle()['summary']['run_status'],'completed')
        self.assertEqual(self.base.step_calls,7)
    def test_fault_hard_constraint_wins_and_new_episode_does_not_reuse_history(self):
        self.warm();self.confirm();before=self.base.step_calls
        self.s.reject_probe();self.assertTrue(self.s.faulted);self.assertEqual(self.base.step_calls,before)
        with self.assertRaises(RuntimeError):self.confirm()
        old=self.s.episode;self.s.restart(operator='test',reason='explicit new simulator episode')
        self.assertNotEqual(old,self.s.episode);self.assertEqual(self.s.samples,0)
        self.assertEqual(self.base.step_calls,before)
        with self.assertRaises(RuntimeError):self.s.cycle()
    def test_unavailable_scope_and_anomaly_latch_at_first_full_window(self):
        self.model.in_scope=False
        for _ in range(4):self.s.sample(operator='test',reason='manual sample')
        self.assertTrue(self.s.faulted)
        with self.assertRaises(RuntimeError):self.s.sample(operator='test',reason='cannot override scope')
        self.assertEqual(self.base.step_calls,4)
    def test_stale_state_does_not_execute_and_confirmation_needs_resources_clear(self):
        self.warm();self.s.system.scheduler.active_agent_tasks['busy']='task'
        with self.assertRaises(RuntimeError):self.confirm()
        self.s.system.scheduler.active_agent_tasks.clear();self.confirm();self.clock[0]=61.
        result=self.s.cycle();self.assertNotEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.base.step_calls,6);self.assertTrue(self.s.faulted)
    def test_duplicate_confirmation_and_repeated_reads_do_not_create_clean_samples(self):
        self.warm();clean=self.s.policy.clean
        for _ in range(10):self.s.status()
        self.assertEqual(self.s.policy.clean,clean)
        self.confirm()
        with self.assertRaises(RuntimeError):self.confirm()
        self.assertEqual(self.base.step_calls,6)
    def test_unknown_step_result_never_retried(self):
        def broken(action):
            self.base.step_calls+=1;self.base._ready=False
            raise RuntimeError('unknown solver result')
        self.base.step=broken
        result=self.s.sample(operator='test',reason='one explicit action')
        self.assertNotEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.base.step_calls,1)
        with self.assertRaises(RuntimeError):self.s.sample(operator='test',reason='no retry')
    def test_changed_model_action_space_invalidates_review(self):
        self.warm();self.confirm();self.base.action_space.high[0]=2.
        result=self.s.cycle()
        self.assertNotEqual(result['summary']['run_status'],'completed')
        self.assertEqual(self.base.step_calls,6);self.assertTrue(self.s.faulted)
    def test_http_credentials_and_repeated_form(self):
        import re
        client=create_app(self.s).test_client();html=client.get('/').get_data(as_text=True)
        data={k:re.search('name="'+k+'" value="([^"]+)"',html).group(1) for k in ('token','revision')}
        data.update(operator='test',reason='explicit sample')
        self.assertEqual(client.post('/sample',data={}).status_code,403)
        self.assertEqual(client.post('/sample',data=data).status_code,303)
        self.assertEqual(client.post('/sample',data=data).status_code,409)
        self.assertEqual(self.base.step_calls,1)

if __name__=='__main__':unittest.main()
