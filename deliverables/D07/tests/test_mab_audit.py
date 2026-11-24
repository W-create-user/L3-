"""Fixture-only audit corruption checks; never counted as actual mAb runs."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from l3_system.monitoring.mab_comparison_audit import audit,ARMS

def write(path,obj):path.write_text(json.dumps(obj)+'\n')

class MAbAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        rows=[];specs=[]
        for condition in ('nominal','sensor_fault','process_disturbance'):
            for seed in range(10):
                pair=f'{condition}-{seed}'
                for arm in ARMS:
                    ident=f'{pair}-{arm}';folder=self.root/ident;folder.mkdir()
                    spec=dict(run_id=ident,pair_id=pair,condition=condition,seed=seed,arm=arm,initial_buffer_offset_dm=0.)
                    specs.append(spec);rows.append({**spec,'started':True,'success':True,'step_calls':3,
                        'initial_internal_state_sha256':hashlib.sha256(pair.encode()).hexdigest(),
                        'pending_requests_clear':True,'C_resources_clear':True,'wall_seconds':.1,'simulated_makespan_min':180.})
                    raw=[0.]*1970;raw[17]=5.
                    write(folder/'steps.json',[dict(raw_preclip_observation=raw,action=[0.]*9,step_count=i,
                        state_version=i+1,success=True,info={'clipped_indices':[19]}) for i in range(1,4)])
                    if arm=='four_agent':write(folder/'A_goal_result.json',{'goal_spec':{'allowed_actions':
                        {f'model_action_{i}':{'range':{'lower':-1.,'upper':1.}} for i in range(9)}}})
        write(self.root/'frozen_plan.json',{'schema_version':'D07MAbPaired/1.0','runs':specs,'source_sha256':{}})
        self.report={'schema_version':'D24MAbSystemComparison/1.0','pilot':False,
            'plan_sha256':hashlib.sha256((self.root/'frozen_plan.json').read_bytes()).hexdigest(),'runs':rows}
        write(self.root/'summary.json',self.report)
    def test_exact_counts_and_zero_improvement_without_production_claim(self):
        r=audit(self.root);self.assertEqual(r['successful_runs'],90);self.assertEqual(r['raw_step_calls'],270)
        self.assertEqual(r['mean_simulated_makespan_reduction'],0)
        self.assertEqual(r['records_with_any_model_clipping'],270)
        self.assertFalse(r['whole_project_90_percent_claim']);self.assertFalse(r['physical_safety_certified'])
    def test_failure_remains_in_started_denominator(self):
        r=self.report['runs'][2];r['success']=False;r['simulated_makespan_min']=None
        write(self.root/'summary.json',self.report);result=audit(self.root)
        self.assertEqual(result['started_runs'],90);self.assertEqual(result['successful_runs'],89)
        self.assertEqual(result['complete_successful_makespan_pairs'],29)
    def test_missing_run_is_rejected(self):
        self.report['runs'].pop();write(self.root/'summary.json',self.report)
        with self.assertRaises(ValueError):audit(self.root)
    def test_raw_state_violation_cannot_be_hidden_by_success_label(self):
        folder=self.root/self.report['runs'][0]['run_id'];steps=json.loads((folder/'steps.json').read_text())
        steps[0]['raw_preclip_observation'][17]=100.;write(folder/'steps.json',steps)
        with self.assertRaises(ValueError):audit(self.root)
    def test_out_of_bounds_action_and_version_gap_are_rejected(self):
        folder=self.root/self.report['runs'][0]['run_id'];steps=json.loads((folder/'steps.json').read_text())
        steps[0]['action'][0]=2.;write(folder/'steps.json',steps)
        with self.assertRaises(ValueError):audit(self.root)
        steps[0]['action'][0]=0.;steps[1]['state_version']+=2;write(folder/'steps.json',steps)
        with self.assertRaises(ValueError):audit(self.root)
    def test_initial_state_mismatch_is_rejected(self):
        self.report['runs'][0]['initial_internal_state_sha256']='0'*64;write(self.root/'summary.json',self.report)
        with self.assertRaises(ValueError):audit(self.root)

if __name__=='__main__':unittest.main()
