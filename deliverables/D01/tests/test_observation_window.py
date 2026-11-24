from copy import deepcopy
import unittest
from l3_system.monitoring.binding import make_binding,check_window
from l3_system.monitoring.observation_window import ObservationWindow

class ObservationWindowTests(unittest.TestCase):
    def setUp(self):
        def var(i):return {'index':i,'name':f'fixture-{i}','unit':'fixture','coordinate':'fixture'}
        self.binding=make_binding({'environment':'Atropine','observation':[var(i) for i in range(39)],'action':[var(i) for i in range(4)]},'1'*64)
        self.collector=ObservationWindow(self.binding);self.collector.start_episode('e1')
    def append(self,v,observation=None,episode_id='e1'):
        return self.collector.append([float(v)]*39 if observation is None else observation,episode_id=episode_id,state_version=v)
    def fill(self):
        for v in range(4):self.append(v)
    def test_warmup_and_bound_packet(self):
        with self.assertRaises(ValueError):self.collector.packet()
        self.fill();packet=self.collector.packet()
        self.assertEqual(packet['state_versions'],[0,1,2,3])
        self.assertEqual(check_window(packet,{'binding':self.binding,'episode_id':'e1','state_version':3},self.binding,4),packet['values'])
    def test_gap_discards_old_context_and_requires_new_window(self):
        self.fill();result=self.append(6)
        self.assertTrue(result['continuity_reset']);self.assertFalse(result['ready'])
        with self.assertRaises(ValueError):self.collector.packet()
        for v in (7,8,9):self.append(v)
        self.assertEqual(self.collector.packet()['state_versions'],[6,7,8,9])
    def test_replay_invalidates_prior_ready_packet(self):
        self.fill()
        with self.assertRaises(ValueError):self.append(3)
        with self.assertRaises(ValueError):self.collector.packet()
        self.assertFalse(self.append(4)['ready'])
    def test_episode_reset_cannot_mix_records(self):
        self.fill();self.collector.start_episode('e2')
        with self.assertRaises(ValueError):self.collector.packet()
        with self.assertRaises(ValueError):self.collector.start_episode('e2')
        with self.assertRaises(ValueError):self.collector.start_episode('e1')
        with self.assertRaises(ValueError):self.append(4)
        self.assertFalse(self.append(0,episode_id='e2')['ready'])
    def test_input_and_output_are_copied(self):
        values=[0.]*39;self.append(0,values);values[0]=999
        for v in (1,2,3):self.append(v)
        packet=self.collector.packet();packet['values'][0][0]=999;packet['binding']['observation'][0]['unit']='changed'
        self.assertEqual(self.collector.packet()['values'][0][0],0)
        self.assertEqual(self.collector.packet()['binding'],self.binding)
    def test_independent_live_context_detects_stale_collector(self):
        self.fill()
        with self.assertRaises(ValueError):check_window(self.collector.packet(),{'binding':self.binding,'episode_id':'e1','state_version':4},self.binding,4)
    def test_invalid_observation_clears_window(self):
        for observation in ([0.]*38,[True]*39,[float('nan')]*39,['1']*39):
            self.collector=ObservationWindow(self.binding);self.collector.start_episode('e1');self.fill()
            with self.assertRaises(ValueError):self.append(4,observation)
            with self.assertRaises(ValueError):self.collector.packet()
    def test_bound_statistical_entry_rejects_warmup_and_independently_stale_context(self):
        from types import SimpleNamespace
        from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
        from l3_system.monitoring.statistical_monitor import StatisticalMonitoringAgent
        adapter=FakeAdapter();live={'binding':self.binding,'episode_id':'e1','state_version':3}
        detector=SimpleNamespace(dimension=39,window_size=4,threshold=.5,label_scope='sensor_fault',binding=self.binding,
                                 score=lambda window:{'anomalous':False})
        monitor=StatisticalMonitoringAgent(adapter.action_space,adapter.observation_space,detector=detector,
            window_provider=self.collector.packet,trusted_context_provider=lambda:live,enable_experimental=True)
        request={'task_id':'fixture','action':[0.]*4,'state_version':3}
        self.assertEqual(monitor.check_request(request,[3.]*39)['decision'],'reject')
        self.fill();self.assertEqual(monitor.check_request(request,[3.]*39)['decision'],'pass')
        live['state_version']=4;request['state_version']=4
        self.assertEqual(monitor.check_request(request,[3.]*39)['decision'],'reject')
        self.assertEqual(adapter.step_calls,0)

    def test_placeholder_binding_rejected(self):
        binding=deepcopy(self.binding);binding['collection_config_sha256']='0'*64
        with self.assertRaises(ValueError):ObservationWindow(binding)

if __name__=='__main__':unittest.main()
