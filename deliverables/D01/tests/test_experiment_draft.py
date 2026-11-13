from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'deliverables/D01/scripts'))
from validate_experiment_draft import validate

class ExperimentDraftTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads((ROOT/'configs/research/november_experiment_protocol.draft.json').read_text())
    def test_draft_has_missing_inputs_and_never_claims_frozen(self):
        result=validate(self.plan)
        self.assertEqual(result['errors'],[]);self.assertTrue(result['missing_confirmations'])
        self.assertFalse(result['ready_to_freeze'])
    def test_declared_confirmations_do_not_authorize_run(self):
        self.plan['required_confirmations']={k:'declared-ref' for k in self.plan['required_confirmations']}
        self.assertFalse(validate(self.plan)['ready_to_freeze'])
        self.plan['execution_authorized']=True
        self.assertTrue(validate(self.plan)['errors'])
    def test_duplicate_or_boolean_seeds_rejected(self):
        for value in (True,26001):
            plan=deepcopy(self.plan);plan['conditions'][0]['pair_seeds'][1]=value
            self.assertTrue(validate(plan)['errors'])
    def test_frozen_status_rejected(self):
        self.plan['status']='frozen';self.assertTrue(validate(self.plan)['errors'])
    def test_non_object_rejected(self):self.assertTrue(validate(None)['errors'])

if __name__=='__main__':unittest.main()
