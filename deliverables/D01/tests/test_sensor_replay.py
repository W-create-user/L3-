"""Synthetic observations only; no received trajectory is used as a fixture."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from sensor_fault_replay import replay,run
from l3_system.agents.safety_rules import observation_integrity_error
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter


class SensorReplayTests(unittest.TestCase):
    def setUp(self):
        def variable(index):
            return {'index':index,'name':f'fixture-{index}','unit':'fixture_unit',
                    'coordinate':'fixture_absolute','source_reference':'synthetic-unit-test-only'}
        self.dictionary={'schema_version':'D01Variables/0.1','environment':'Atropine',
                         'observation':[variable(i) for i in range(39)],'action':[variable(i) for i in range(4)],
                         'review':{'reviewed_by':'automated-fixture','reviewed_on':'2026-11-12',
                                   'source_reference':'not a physical dictionary'}}
        self.rows=[{'schema_version':'D01Step/0.1','episode_id':'synthetic-fixture','step_index':i,
                    'simulation_time':i*600,'simulation_time_unit':'s','collected_at':f'2026-11-12T06:0{i}:00Z',
                    'observation_before':[float(i)]*39,'observation_after':[float(i+1)]*39,
                    'candidate_action':[0.0]*4,'executed_action':[0.0]*4,'executed':True,
                    'state_version_before':i+1,'state_version_after':i+2,'reward':0.0,
                    'terminated':False,'truncated':False,'execution_status':'completed','failure_reason':None}
                   for i in range(5)]
        self.spec={'schema_version':'SensorReplay/0.1','mode':'offline_only','environment':'Atropine',
                   'source_episode_id':'synthetic-fixture','source_lineage_group':'fixture-root',
                   'source_split':'excluded','source_sha256':'1'*64,'dictionary_sha256':'2'*64,
                   'events':[self.event('missing')]}

    def event(self,kind,**kwargs):
        parameters={'missing':{},'stuck':{'reference_record':0},'bias':{'offset':2.0},
                    'drift':{'increment_per_record':0.5}}[kind]
        return dict(event_id='fixture-event',kind=kind,variable_index=6,unit='fixture_unit',
                    coordinate='fixture_absolute',start_record=1,end_record=3,parameters=parameters,**kwargs)

    def output(self):return replay(self.rows,self.spec,self.dictionary)

    def test_missing_is_nullable_and_integrity_rejected(self):
        result=self.output()
        self.assertEqual([r['observation'][6] for r in result['replay_inputs']],[0.0,None,None,None,4.0])
        self.assertEqual(result['summary']['replay_integrity_rejections'],3)
        self.assertFalse(result['replay_inputs'][1]['available_mask'][6])
        json.dumps(result,allow_nan=False)

    def test_bias_remains_finite_and_is_not_detected_by_integrity(self):
        self.spec['events']=[self.event('bias')];result=self.output()
        self.assertEqual([r['observation'][6] for r in result['replay_inputs']],[0,3,4,5,4])
        self.assertEqual(result['summary']['replay_integrity_rejections'],0)
        self.assertEqual(result['summary']['fault_active_integrity_passes'],3)
        self.assertFalse(result['summary']['process_anomaly_predictions_available'])

    def test_drift_per_record_starts_at_first_increment(self):
        self.spec['events']=[self.event('drift')]
        self.assertEqual([r['observation'][6] for r in self.output()['replay_inputs']],[0,1.5,3,4.5,4])

    def test_stuck_uses_clean_reference_and_recovers(self):
        self.spec['events']=[self.event('stuck')]
        self.assertEqual([r['observation'][6] for r in self.output()['replay_inputs']],[0,0,0,0,4])

    def test_stuck_can_be_active_without_changed_value(self):
        self.spec['events']=[self.event('stuck')];self.spec['events'][0]['parameters']['reference_record']=1
        result=self.output()
        self.assertTrue(result['labels'][1]['sensor_fault_active'])
        self.assertFalse(result['integrity_checks'][1]['measurement_changed'])

    def test_clean_source_and_spec_never_mutated(self):
        before=deepcopy((self.rows,self.spec,self.dictionary));self.output()
        self.assertEqual((self.rows,self.spec,self.dictionary),before)

    def test_labels_separate_from_features_and_no_process_truth(self):
        result=self.output()
        for row in result['replay_inputs']:
            self.assertEqual(set(row),{'sample_index','observation','available_mask'})
        for label in result['labels']:self.assertIsNone(label['process_anomaly_label'])

    def test_multiple_variables_can_overlap(self):
        second=self.event('bias');second.update(event_id='other',variable_index=37)
        self.spec['events'].append(second)
        self.assertEqual(len(self.output()['labels'][1]['active_event_ids']),2)

    def test_overlapping_same_variable_rejected(self):
        second=self.event('bias');second['event_id']='other';self.spec['events'].append(second)
        with self.assertRaises(ValueError):self.output()

    def test_future_stuck_reference_rejected(self):
        self.spec['events']=[self.event('stuck')];self.spec['events'][0]['parameters']['reference_record']=4
        with self.assertRaises(ValueError):self.output()

    def test_wrong_unit_coordinate_or_environment_rejected(self):
        for field in ('unit','coordinate'):
            spec=deepcopy(self.spec);spec['events'][0][field]='wrong'
            with self.assertRaises(ValueError):replay(self.rows,spec,self.dictionary)
        self.spec['environment']='mAb'
        with self.assertRaises(ValueError):self.output()

    def test_different_source_episode_or_order_rejected(self):
        self.rows[0]['episode_id']='other'
        with self.assertRaises(ValueError):self.output()
        self.rows[0]['episode_id']='synthetic-fixture';self.rows[1]['step_index']=0
        with self.assertRaises(ValueError):self.output()

    def test_invalid_windows_and_indices_rejected(self):
        for field,value in [('start_record',-1),('end_record',5),('variable_index',39),('variable_index',True)]:
            spec=deepcopy(self.spec);spec['events'][0][field]=value
            with self.assertRaises(ValueError):replay(self.rows,spec,self.dictionary)

    def test_duplicate_event_ids_rejected(self):
        self.spec['events'].append(self.event('bias'))
        with self.assertRaises(ValueError):self.output()

    def test_zero_boolean_and_nonfinite_amplitudes_rejected(self):
        for value in (0,True,float('nan'),float('inf')):
            self.spec['events']=[self.event('bias')];self.spec['events'][0]['parameters']['offset']=value
            with self.assertRaises(ValueError):self.output()

    def test_injection_overflow_rejected_before_output(self):
        self.spec['events']=[self.event('bias')];self.spec['events'][0]['parameters']['offset']=1e308
        self.rows[1]['observation_before'][6]=1e308
        with self.assertRaises(ValueError):self.output()

    def test_placeholder_hashes_and_live_mode_rejected(self):
        self.spec['source_sha256']='0'*64
        with self.assertRaises(ValueError):self.output()
        self.spec['source_sha256']='1'*64;self.spec['mode']='live'
        with self.assertRaises(ValueError):self.output()

    def test_process_label_cannot_be_supplied_in_spec(self):
        self.spec['process_anomaly_label']=True
        with self.assertRaises(ValueError):self.output()

    def test_source_must_be_clean(self):
        self.rows[0]['observation_before'][6]=None
        with self.assertRaises(ValueError):self.output()

    def test_shared_integrity_predicate_matches_monitor(self):
        adapter=FakeAdapter();monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
        for observation in ([0.0]*39,[None]*39,[float('nan')]*39,[0.0]*38):
            self.assertEqual(observation_integrity_error(observation,(39,)),monitor.observation_error(observation))

    def file_fixture(self,root):
        source=root/'source.jsonl';dictionary=root/'dictionary.json';spec=root/'spec.json'
        source.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        dictionary.write_text(json.dumps(self.dictionary))
        self.spec.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         dictionary_sha256=hashlib.sha256(dictionary.read_bytes()).hexdigest())
        spec.write_text(json.dumps(self.spec))
        return source,dictionary,spec

    def test_file_roundtrip_preserves_lineage_split_source_and_no_simulator_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dictionary,spec=self.file_fixture(root);original=source.read_bytes()
            with (patch('l3_system.execution.environment_adapter.EnvironmentAdapter.step',side_effect=AssertionError('step forbidden')) as step,
                  patch('l3_system.execution.environment_adapter.EnvironmentAdapter.reset',side_effect=AssertionError('reset forbidden')) as reset):
                summary=run(source,dictionary,spec,root/'output')
            step.assert_not_called();reset.assert_not_called()
            self.assertEqual(source.read_bytes(),original);self.assertEqual(summary['new_simulator_calls'],0)
            provenance=json.loads((root/'output/provenance.json').read_text())
            self.assertEqual(provenance['source_lineage_group'],provenance['derived_lineage_group'])
            self.assertEqual(provenance['source_split'],provenance['derived_split'])
            self.assertFalse(provenance['accepted_for_training'])
            for name,expected in provenance['output_sha256'].items():
                self.assertEqual(hashlib.sha256((root/'output'/name).read_bytes()).hexdigest(),expected)

    def test_hash_mismatch_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dictionary,spec=self.file_fixture(root)
            source.write_text(source.read_text()+'\n')
            with self.assertRaises(ValueError):run(source,dictionary,spec,root/'output')
            self.assertFalse((root/'output').exists())

    def test_output_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dictionary,spec=self.file_fixture(root);output=root/'output';output.mkdir()
            (output/'keep').write_text('existing')
            with self.assertRaises(FileExistsError):run(source,dictionary,spec,output)
            self.assertEqual((output/'keep').read_text(),'existing')

    def test_cli_offline_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,dictionary,spec=self.file_fixture(root)
            result=subprocess.run([sys.executable,str(ROOT/'deliverables/D01/scripts/sensor_fault_replay.py'),
                                   '--source',str(source),'--dictionary',str(dictionary),'--spec',str(spec),
                                   '--output',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)['R03_enabled'])


if __name__=='__main__':unittest.main()
