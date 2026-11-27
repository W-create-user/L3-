import copy
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from l3_system.execution import ExecutionAgent
from l3_system.execution.joint_action_gateway import JointActionGateway
from l3_system.execution.training_wrapper import JointActionResearchEnv
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter

class OwnedFake(FakeAdapter):
    def __init__(self):
        super().__init__();self.metadata={'environment_id':'TEST_COUPLED','action_coordinate':'test_model','episode_id':'unreset'};self.episodes=0
    def reset(self,seed=1):
        super().reset(seed);self.episodes+=1;self.state_version+=1;self.step_count=0
        self.metadata['episode_id']='fake-episode-'+str(self.episodes)
        return self.get_observation()

CONFIG={'environment_id':'TEST_COUPLED','action_coordinate':'test_model','action_size':4,'observation_size':39,
        'partitions':[{'actor_id':'left','action_indices':[0,1],'observation_indices':[0,1]},
                      {'actor_id':'right','action_indices':[2,3],'observation_indices':[2,3]}]}

class JointGatewayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.a=OwnedFake();self.e=ExecutionAgent(self.a,log_path=Path(self.tmp.name)/'B.jsonl');self.e.reset()
        self.m=MonitoringAgent(self.a.action_space,self.a.observation_space);self.now=[0.]
        self.g=JointActionGateway(self.e,self.m,CONFIG,clock=lambda:self.now[0])
    def begin(self,name='cycle'):return self.g.begin(name,timeout_seconds=1)
    def submit(self,context,actor,action):return self.g.submit(context['cycle_id'],actor,action,state_version=context['state_version'],episode_id=context['episode_id'],coordinate=context['coordinate'])
    def fill(self,context):
        self.submit(context,'right',[.2,.3]);self.submit(context,'left',[.0,.1])
    def test_out_of_order_components_and_concurrent_finish_step_once(self):
        c=self.begin();self.fill(c)
        def worker_finish(_):
            try:return self.g.finish('cycle')
            except RuntimeError:return None
        with ThreadPoolExecutor(max_workers=4) as workers:blocked=list(workers.map(worker_finish,range(4)))
        self.assertTrue(all(r is None for r in blocked));self.assertEqual(self.a.step_calls,0)
        expected=self.g.finish('cycle')
        with ThreadPoolExecutor(max_workers=4) as workers:results=list(workers.map(lambda _:self.g.finish('cycle'),range(4)))
        self.assertEqual(self.a.step_calls,1);self.assertTrue(all(r==expected for r in results))
        self.assertEqual(results[0]['executed_joint_action'],[0.,.1,.2,.3]);self.assertFalse(results[0]['learning_eligible'])
        with self.assertRaises(ValueError):self.begin('cycle')
    def test_missing_timeout_and_late_component_never_fill_defaults(self):
        c=self.begin();self.submit(c,'left',[0,0]);self.assertEqual(self.g.finish('cycle')['status'],'waiting')
        self.now[0]=2;r=self.g.finish('cycle');self.assertEqual(r['status'],'timeout');self.assertEqual(self.a.step_calls,0);self.assertIsNone(r['reward'])
        with self.assertRaises(RuntimeError):self.submit(c,'right',[0,0])
    def test_duplicate_taints_round_without_execution(self):
        c=self.begin();self.submit(c,'left',[0,0])
        with self.assertRaises(ValueError):self.submit(c,'left',[0,0])
        self.assertEqual(self.g.finish('cycle')['status'],'invalid_component');self.assertEqual(self.a.step_calls,0)
    def test_reset_stales_pending_components(self):
        c=self.begin();self.fill(c);self.e.reset(seed=2)
        r=self.g.finish('cycle');self.assertEqual(r['status'],'stale_round');self.assertIsNone(r['observation_after']);self.assertEqual(self.a.step_calls,0)
    def test_d_rejection_has_no_action_reward_or_training_transition(self):
        c=self.begin();self.submit(c,'left',[2,0]);self.submit(c,'right',[0,0]);r=self.g.finish('cycle')
        self.assertTrue(r['rejected']);self.assertFalse(r['executed']);self.assertIsNone(r['executed_joint_action']);self.assertIsNone(r['reward'])
        self.assertEqual(self.a.step_calls,0);self.assertTrue(self.g.takeover_required)
    def test_unknown_physical_failure_latches_until_human_ack_after_reset(self):
        c=self.begin();self.fill(c)
        def broken(action):self.a.step_calls+=1;self.a._ready=False;raise RuntimeError('uncertain solver return')
        self.a.step=broken;r=self.g.finish('cycle');self.assertIsNone(r['executed']);self.assertEqual(r['step_delta'],1)
        with self.assertRaises(RuntimeError):self.begin('next')
        self.e.reset(seed=3);self.g.acknowledge_after_reset(operator='offline-test',reason='explicit new simulation episode')
        self.assertFalse(self.g.takeover_required);self.assertEqual(self.a.step_calls,1)
    def test_only_one_gateway_owns_an_adapter(self):
        with self.assertRaises(RuntimeError):JointActionGateway(self.e,self.m,CONFIG)
    def test_overlapping_or_missing_partitions_are_rejected(self):
        for indices in [[1,2],[2]]:
            a=OwnedFake();e=ExecutionAgent(a,log_path=Path(self.tmp.name)/str(len(indices)));e.reset()
            bad=copy.deepcopy(CONFIG);bad['partitions'][1]['action_indices']=indices
            with self.assertRaises(ValueError):JointActionGateway(e,self.m,bad)
    def test_training_requires_explicit_research_scope(self):
        with self.assertRaises(RuntimeError):JointActionResearchEnv(self.g)
        w=JointActionResearchEnv(self.g,research_only=True)
        r=w.step({'right':[0,0],'left':[0,0]});self.assertTrue(r['transition_complete']);self.assertFalse(r['learning_eligible'])
    def test_wrong_message_context_and_nonfinite_components_do_not_step(self):
        for index,value in enumerate([[float('nan'),0],[0],[True,False]]):
            c=self.begin('bad'+str(index))
            with self.assertRaises(ValueError):self.submit(c,'left',value)
        self.assertEqual(self.a.step_calls,0)

    def test_deadline_rechecked_after_D_review_before_step(self):
        context=self.begin();self.fill(context);original=self.m.check_request
        def delayed(request,observation):
            reply=original(request,observation);self.now[0]=2;return reply
        self.m.check_request=delayed
        result=self.g.finish('cycle')
        self.assertFalse(result['success']);self.assertEqual(self.a.step_calls,0)

    def test_wrong_coordinate_is_not_guessed_from_numeric_values(self):
        context=self.begin()
        with self.assertRaisesRegex(ValueError,'coordinate'):
            self.g.submit('cycle','left',[0,0],state_version=context['state_version'],episode_id=context['episode_id'],coordinate='physical_or_normalized_guess')
        self.assertEqual(self.a.step_calls,0)

    def test_parallel_component_submissions_are_collected_before_one_owner_step(self):
        context=self.begin()
        with ThreadPoolExecutor(max_workers=2) as workers:
            results=list(workers.map(lambda actor:self.submit(context,actor,[0,0]),['right','left']))
        self.assertTrue(all(row['status']=='accepted' for row in results));self.assertEqual(self.a.step_calls,0)
        self.assertTrue(self.g.finish('cycle')['success']);self.assertEqual(self.a.step_calls,1)
