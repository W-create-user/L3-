"""Synthetic contract fixtures; received package is separately reviewed as evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_b_handoff import make_dictionary, disabled_rules, normalize_result, verify_hashes


class BAdapterTests(unittest.TestCase):
    def setUp(self):
        self.schema=dict(backend='AtropineEnvGym',dimension=39,normalize=False,
                         fields=[dict(index=i,name=f'fixture_{i}',unit='fixture_unit') for i in range(39)])
        self.actions=[dict(backend='AtropineEnvGym',index=i,name=f'fixture_action_{i}',unit='fixture_unit',
                           normalize=False,request_scale='delta_from_USS') for i in range(4)]
        self.actions += [dict(backend='MAbEnvGym',index=i,name=f'mab_fixture_{i}',unit='unconfirmed') for i in range(9)]
        self.request=dict(request_id='R',task_id='T',environment='SMPL.AtropineEnvGym',normalize=False,
                          uss_subtracted=True,observation=[0]*39,action=[0.01]*4,state_version=1)
        self.result=dict(request_id='R',task_id='T',observation=[0.1]*39,action=[0.01]*4,executed=True,
                         state_version=2,reward=0.0,terminated=False,truncated=False,
                         execution_status='completed',failure_reason=None,step_count=1)
        self.row=dict(run_id='fixture',previous_observation=[0]*39,requested_action=[0.01]*4,
                      actual_executed_action=[0.01]*4,executed=True,result=self.result,sim_time_min=10,step_count=1)
        self.raw={'time':'2026-11-12T10:00:00+08:00'}

    def dictionary(self):return make_dictionary(self.schema,self.actions,'1'*40)

    def test_environment_spaces_not_mixed(self):
        dictionary=self.dictionary();self.assertEqual(len(dictionary['observation']),39)
        self.assertEqual(len(dictionary['action']),4)
        self.assertFalse(any('mab_fixture' in v['name'] for v in dictionary['action']))

    def test_mab_cannot_use_atropine_adapter(self):
        self.schema.update(backend='MAbEnvGym',dimension=1970)
        with self.assertRaises(ValueError):self.dictionary()

    def test_different_action_coordinates_rejected(self):
        self.actions[0]['request_scale']='absolute'
        with self.assertRaises(ValueError):self.dictionary()

    def test_uncertain_units_not_rewritten_as_confirmed(self):
        self.schema['fields'][7]['unit']='model_coordinate_unspecified'
        dictionary=self.dictionary()
        self.assertEqual(dictionary['observation'][7]['unit'],'model_coordinate_unspecified')
        self.assertIn('not physical-unit approval',dictionary['review']['reviewed_by'])

    def test_disabled_rules_have_no_thresholds(self):
        config=disabled_rules(self.dictionary())
        self.assertFalse(config['enabled']);self.assertEqual(config['rules'],[])
        self.assertEqual(config['observation_shape'],[39]);self.assertEqual(config['action_shape'],[4])

    def test_request_action_keeps_delta_coordinates(self):
        row=normalize_result(self.row,self.request,self.raw,0)
        self.assertEqual(row['executed_action'],[0.01]*4)
        self.assertEqual(row['simulation_time'],600)
        self.assertEqual(row['observation_before'][0],0)
        self.assertNotIn('process_anomaly_label',row)

    def test_rejected_raw_observation_not_forged_as_step_output(self):
        self.result.update(executed=False,execution_status='rejected',reward=None,failure_reason='R-02',
                           observation=[0]*39,state_version=1,step_count=0)
        self.row.update(executed=False,actual_executed_action=None,sim_time_min=0,step_count=0)
        converted=normalize_result(self.row,self.request,self.raw,0)
        self.assertIsNone(converted['observation_after']);self.assertIsNone(converted['executed_action'])
        self.assertEqual(self.result['observation'],[0]*39)

    def test_modified_previous_observation_rejected(self):
        self.row['previous_observation']=[9]*39
        with self.assertRaises(ValueError):normalize_result(self.row,self.request,self.raw,0)

    def test_wrong_task_and_time_rejected(self):
        self.result['task_id']='other'
        with self.assertRaises(ValueError):normalize_result(self.row,self.request,self.raw,0)
        self.result['task_id']='T';self.row['sim_time_min']=0
        with self.assertRaises(ValueError):normalize_result(self.row,self.request,self.raw,0)

    def test_hashes_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack=Path(tmp);(pack/'a.txt').write_text('fixture')
            (pack/'manifest.json').write_text(json.dumps({'algorithm':'SHA256','files':{
                'a.txt':hashlib.sha256(b'fixture').hexdigest()}}))
            self.assertEqual(verify_hashes(pack),1)
            (pack/'a.txt').write_text('changed')
            with self.assertRaises(ValueError):verify_hashes(pack)


if __name__=='__main__':unittest.main()
