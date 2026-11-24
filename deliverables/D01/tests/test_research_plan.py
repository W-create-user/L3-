"""Guard the draft against accidental threshold/label/collection approval."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from validate_research_plan import read,validate


class ResearchPlanTests(unittest.TestCase):
    def setUp(self):
        self.catalog=read(ROOT/'configs/research/atropine_rule_candidates.draft.json')
        self.protocol=read(ROOT/'configs/research/atropine_collection_protocol.draft.json')

    def bad(self):self.assertTrue(validate(self.catalog,self.protocol))

    def test_published_draft_is_consistent_but_disabled(self):
        self.assertEqual(validate(self.catalog,self.protocol),[])
        self.assertFalse(self.protocol['execution_authorized'])

    def test_no_implicit_collection_approval(self):
        self.protocol['execution_authorized']=True;self.bad()

    def test_no_enabled_candidate(self):
        self.catalog['candidates'][0]['enabled']=True;self.bad()

    def test_no_unverified_limit(self):
        self.catalog['candidates'][0]['threshold']['upper']=20;self.bad()

    def test_no_backend_reuse(self):
        self.protocol['environment']='MAbEnvGym';self.bad()

    def test_no_coordinate_rescaling(self):
        self.catalog['candidates'][3]['coordinate']='absolute_plant_input';self.bad()

    def test_no_wrong_variable_index(self):
        self.catalog['candidates'][0]['index']=0;self.bad()

    def test_no_duplicate_rule(self):
        self.catalog['candidates'][1]=deepcopy(self.catalog['candidates'][0]);self.bad()

    def test_evidence_required(self):
        self.catalog['candidates'][0]['evidence']=[];self.bad()

    def test_interface_rejection_is_not_process_truth(self):
        self.protocol['scenario_families'][0]['process_anomaly_label']=True;self.bad()

    def test_sensor_replay_is_not_closed_loop(self):
        self.protocol['scenario_families'][2]['mode']='live_closed_loop';self.bad()

    def test_process_injection_point_not_invented(self):
        self.protocol['scenario_families'][3]['injection_site']='arbitrary_temperature';self.bad()

    def test_splits_follow_original_lineage(self):
        self.protocol['split_policy']['derived_variants_follow_source_group']=False;self.bad()

    def test_seed_not_proof_of_independence(self):
        self.protocol['split_policy']['seed_alone_is_independence_evidence']=True;self.bad()

    def test_no_auto_normal_labels(self):
        self.protocol['pilot_budget_proposal']['normal_label_auto_assign']=True;self.bad()

    def test_phase_budget_matches_total(self):
        self.protocol['pilot_budget_proposal']['cycles_per_base_episode']=100;self.bad()

    def test_simulation_period_not_freshness_budget(self):
        self.protocol['clock_policy']['freshness_limit_seconds']=600;self.bad()

    def test_guards_not_disabled_for_experiments(self):
        self.protocol['stop_policy']['keep_R01_R02_and_runtime_guards']=False;self.bad()

    def test_no_model_defined_ground_truth(self):
        self.protocol['label_contract']['preflight_and_model_verdict_cannot_define_ground_truth']=False;self.bad()

    def test_no_claims_from_draft(self):
        self.protocol['scientific_claims_allowed']=True;self.bad()


if __name__=='__main__':unittest.main()
