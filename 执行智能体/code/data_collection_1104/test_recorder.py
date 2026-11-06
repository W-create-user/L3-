import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from .recorder import ExperimentRecorder
from .validate_data import validate_episode


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'episode'
        self.before = {'observation':[1.0,2.0], 'step_count':0,
                       'step_calls':0,'state_version':1,'done':False}

    def recorder(self):
        recorder = ExperimentRecorder(self.directory, run_id='unit',episode_id='normal',
                                      observation_dim=2,action_dim=2,dt_minutes=10)
        recorder.record_initial(self.before, {'seed':1101,'state':copy.deepcopy(self.before)})
        (self.directory / 'main.jsonl').write_text('{}\n',encoding='utf-8')
        return recorder

    def transition(self, before=None, *, rejected=False, reward=-0.5, request_id='R1'):
        before = copy.deepcopy(before or self.before)
        after = copy.deepcopy(before)
        if not rejected:
            after.update(observation=[1.5,2.5],step_count=before['step_count']+1,
                         step_calls=before['step_calls']+1,state_version=before['state_version']+1)
        request = {'request_id':request_id,'task_id':'任务001','observation':before['observation'],
                   'step_count':before['step_count'],'action':[0.2,-0.1]}
        result = {'request_id':request_id,'task_id':'任务001','subtask_id':'SUB1',
                  'action':[0.2,-0.1],'observation':after['observation'],
                  'execution_status':'rejected' if rejected else 'completed',
                  'executed':not rejected,'success':not rejected,'reward':None if rejected else reward,
                  'done':False,'step_count':after['step_count'],
                  'failure_reason':'越界，拒绝\n请重新生成' if rejected else None,
                  'safety_decision':{'approved':not rejected,'rule_id':'R-02' if rejected else None,
                                     'risk_level':3 if rejected else 0}}
        return {'action_request':request,'execution_result':result},before,after

    def write_transition(self, recorder, trace, before, after):
        recorder.record_attempt(trace,before,after,wall_duration_ms=12.5,main_trace_file='main.jsonl')

    def test_normal_and_rejection_have_two_attempts_but_one_step(self):
        with self.recorder() as recorder:
            trace,before,after = self.transition()
            self.write_transition(recorder,trace,before,after)
            trace2,before2,after2 = self.transition(after,rejected=True,request_id='R2')
            self.write_transition(recorder,trace2,before2,after2)
            summary=recorder.finish(requested_steps=1,stop_reason='done')
        validated=validate_episode(self.directory)
        self.assertEqual(validated['status'],'PASS')
        self.assertEqual(summary['actual_step_calls'],1)
        self.assertEqual(summary['observation_rows'],2)
        with (self.directory/'steps.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['reward'],'-0.5')
        self.assertEqual(rows[1]['reward'],'')
        self.assertEqual(rows[1]['failure_reason'],'越界，拒绝\n请重新生成')
        self.assertEqual(rows[1]['nominal_sim_time_before_min'],rows[1]['nominal_sim_time_after_min'])

    def test_zero_reward_is_not_lost(self):
        with self.recorder() as recorder:
            self.write_transition(recorder,*self.transition(reward=0.0))
            recorder.finish(requested_steps=1,stop_reason='done')
        self.assertEqual(validate_episode(self.directory)['status'],'PASS')
        with (self.directory/'steps.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['reward'],'0.0')

    def test_duplicate_request_cannot_add_duplicate_row(self):
        with self.recorder() as recorder:
            args=self.transition()
            self.write_transition(recorder,*args)
            with self.assertRaises(ValueError):
                self.write_transition(recorder,*args)
            recorder.finish(requested_steps=1,stop_reason='done')
        self.assertEqual(validate_episode(self.directory)['attempt_rows'],1)

    def test_false_executed_flag_is_detected_before_writing(self):
        with self.recorder() as recorder:
            trace,before,after=self.transition()
            trace['execution_result']['executed']=False
            with self.assertRaises(ValueError):
                self.write_transition(recorder,trace,before,after)
            recorder.finish(requested_steps=0,stop_reason='test')
        self.assertEqual(validate_episode(self.directory)['attempt_rows'],0)

    def test_csv_tampering_is_detected_by_raw_log(self):
        with self.recorder() as recorder:
            self.write_transition(recorder,*self.transition())
            recorder.finish(requested_steps=1,stop_reason='done')
        path=self.directory/'steps.csv'
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            fields=reader.fieldnames
            rows=list(reader)
        rows[0]['reward']='999'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        with self.assertRaises(ValueError):
            validate_episode(self.directory)

    def test_mutating_input_cannot_overwrite_saved_initial_state(self):
        with self.recorder() as recorder:
            self.before['observation'][0]=99
            recorder.finish(requested_steps=0,stop_reason='test')
        initial=json.loads((self.directory/'initial_conditions.json').read_text(encoding='utf-8'))
        self.assertEqual(initial['state']['observation'][0],1.0)
        self.assertEqual(validate_episode(self.directory)['status'],'PASS')

    def test_nonfinite_diagnostic_values_are_explicit_not_invalid_json(self):
        self.before['observation'][0]=float('nan')
        with self.recorder() as recorder:
            recorder.finish(requested_steps=0,stop_reason='diagnostic')
        text=(self.directory/'initial_conditions.json').read_text(encoding='utf-8')
        self.assertIn('__nonfinite__',text)
        self.assertEqual(validate_episode(self.directory)['status'],'PASS')

    def test_exception_keeps_real_attempt_count_without_inventing_reward(self):
        with self.recorder() as recorder:
            after=copy.deepcopy(self.before)
            after.update(step_calls=1,state_version=2)
            recorder.record_exception(RuntimeError('solver error'),before=self.before,after=after,
                                      main_trace_file='main.jsonl',traceback_text='example traceback')
            summary=recorder.finish(requested_steps=1,stop_reason='exception')
        self.assertEqual(summary['experiment_status'],'stopped_early')
        self.assertEqual(summary['actual_step_calls'],1)
        self.assertEqual(validate_episode(self.directory)['exceptions'],1)


if __name__ == '__main__':
    unittest.main()
