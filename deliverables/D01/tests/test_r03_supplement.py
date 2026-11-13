"""Synthetic structural fixtures only; never vendor or upload B trajectories."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import review_r03_supplement as module

class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.pack=self.root/'package';self.pack.mkdir()
        (self.root/'experiments').mkdir();(self.root/'experiments/collect_r03_sensor_evidence.py').write_text('synthetic collector fixture\n')
        (self.root/'source.py').write_text('synthetic source fixture\n')
        self.rule={'runtime_enabled':False,'disposition_proposal':{'approved':False},'backend':'AtropineEnvGym',
            'layout':'atropine-default39-v1','observation_dimension':39,'candidate_relative_limit':.1,
            'variables':{'current':{'index':6,'normalized':False,'steady_state_subtracted':False},
                         'reference':{'index':5,'normalized':False,'steady_state_subtracted':False}},
            'applicable_config':{'normalize':False},'source_evidence':{'fixture':{'path':'source.py','file_sha256':module.digest(self.root/'source.py')}}}
        self.write('rule_candidate.json',self.rule)
        cases=[]
        for value,status in ((13.65,'within_candidate_band'),(14.43,'candidate_breach')):
            observation=[0.]*39;observation[5]=13.;observation[6]=value
            cases.append(dict(observation=observation,origin='manually_constructed_state_not_real_trajectory',
                expected_candidate_status=status,actual={'runtime_enabled':False,'operational_decision':None}))
        self.write('validation_cases.json',cases)
        self.data={'physical_process_labels_available':False,'physical_process_iforest_ready':False,'horizon_steps':30,
            'sampling_period_min':10,'collector_sha256':module.digest(self.root/'experiments/collect_r03_sensor_evidence.py'),'episodes':[]}
        for split,seeds in (('train',range(1301,1307)),('validation',range(1401,1404)),('test',range(1501,1504))):
            for seed in seeds:
                eid=f'{split}_{seed}';name=eid+'/trajectory.jsonl';rows=[]
                for step in range(31):
                    raw=[seed/10000+step/1000]*39;raw[0]=step;delivered=raw.copy()
                    active=split!='train' and 10<=step<=20
                    if active:delivered[6]*=1.2
                    row=dict(episode_id=eid,step=step,sim_time_min=step*10,environment_observation=raw,
                        delivered_observation=delivered,sensor_fault_label=int(active),process_health_label=None)
                    if step==0:row['kind']='reset'
                    else:row.update(done=False,terminated=False,truncated=False,source_step_count=step,
                        requested_action=[0.]*4,actual_action=[0.]*4,label_source='declared_injection_schedule_not_threshold_or_reward')
                    rows.append(row)
                path=self.pack/'sensor_benchmark'/name;path.parent.mkdir(parents=True)
                path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                self.data['episodes'].append(dict(episode_id=eid,group_id=eid,split=split,seed=seed,
                    initial_internal_state_sha256=hashlib.sha256(eid.encode()).hexdigest(),complete=True,reset_count=1,
                    recorded_steps=30,requested_steps=30,process_health_label=None,raw_trajectory=name,trajectory_sha256=module.digest(path),
                    fault=None if split=='train' else dict(target='delivered_observation[6]',type='multiplicative_sensor_bias',multiplier=1.2,first_step=10,last_step=20)))
        self.write('sensor_benchmark/episodes.json',self.data)

    def write(self,name,obj):
        path=self.pack/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj)+'\n')

    def review(self):
        files={str(p.relative_to(self.pack)):module.digest(p) for p in self.pack.rglob('*') if p.is_file() and p.name!='manifest.json'}
        self.write('manifest.json',{'algorithm':'SHA256','files':files})
        with patch.object(module,'EXPECTED_MANIFEST_SHA256',module.digest(self.pack/'manifest.json')):
            return module.review(self.pack,self.root)

    def test_scoped_receipt_and_projection(self):
        episodes,receipt=self.review()
        self.assertEqual(len(episodes),12);self.assertEqual(len(episodes[0]['observations']),30)
        self.assertEqual(len(episodes[0]['observations'][0]),38)
        self.assertNotEqual(episodes[0]['observations'][0][0],1)
        self.assertEqual(receipt['sensor_fault_positive_steps'],66)
        self.assertEqual(receipt['label_scope'],'sensor_fault');self.assertFalse(receipt['deployment_authorized'])

    def test_pin_cannot_silently_change(self):
        self.review()
        with self.assertRaisesRegex(ValueError,'fixed reviewed commit'):module.review(self.pack,self.root)

    def test_physical_rule_cannot_be_enabled(self):
        self.rule['runtime_enabled']=True;self.write('rule_candidate.json',self.rule)
        with self.assertRaises(ValueError):self.review()

    def test_fixed_source_mismatch_rejected(self):
        (self.root/'source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source evidence'):self.review()

    def test_group_leak_rejected(self):
        self.data['episodes'][6]['group_id']=self.data['episodes'][0]['group_id']
        self.write('sensor_benchmark/episodes.json',self.data)
        with self.assertRaisesRegex(ValueError,'lineage'):self.review()

    def test_physical_label_cannot_be_inferred(self):
        self.data['episodes'][0]['process_health_label']='normal'
        self.write('sensor_benchmark/episodes.json',self.data)
        with self.assertRaises(ValueError):self.review()

    def test_changed_schedule_rejected(self):
        self.data['episodes'][6]['fault']['multiplier']=1.3
        self.write('sensor_benchmark/episodes.json',self.data)
        with self.assertRaisesRegex(ValueError,'schedule'):self.review()

    def test_boundary_and_censoring(self):
        obs=[0.]*39;obs[5]=13.;obs[6]=1.1*obs[5]
        self.assertEqual(module.candidate_status(obs),'within_candidate_band')
        obs[6]+=1e-6;self.assertEqual(module.candidate_status(obs),'candidate_breach')
        obs[5]=20.;self.assertEqual(module.candidate_status(obs),'unsupported_reference_or_censoring')
        obs[5]=0.;self.assertEqual(module.candidate_status(obs),'unsupported_reference_or_censoring')
        obs[6]=float('nan')
        with self.assertRaises(ValueError):module.candidate_status(obs)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):module.safe_path(self.root,'../outside')

if __name__=='__main__':unittest.main()
