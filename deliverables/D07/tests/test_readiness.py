import copy
import json
from pathlib import Path
import unittest
from l3_system.monitoring.experiment_readiness import prepare

ROOT=Path(__file__).resolve().parents[3]

class ReadinessTests(unittest.TestCase):
    def plan(self):
        p=json.loads((ROOT/'configs/research/november_experiment_protocol.draft.json').read_text())
        p['run_order_seed']=261123
        return p

    def test_pairing_and_random_order_reproduce_without_launch(self):
        p=self.plan();result=prepare(p,ROOT)
        self.assertEqual(result,prepare(p,ROOT))
        rows=result['ledger'];self.assertEqual(len(rows),90)
        self.assertEqual(len({r['run_id'] for r in rows}),90)
        pairs={}
        for r in rows:
            pairs.setdefault(r['pair_id'],set()).add(r['arm'])
            self.assertFalse(r['started']);self.assertFalse(r['execution_authorized'])
        self.assertEqual(len(pairs),30)
        self.assertTrue(all(v==set(p['arms']) for v in pairs.values()))
        altered=copy.deepcopy(p);altered['run_order_seed']+=1
        self.assertNotEqual(rows,prepare(altered,ROOT)['ledger'])

    def test_missing_and_outside_references_block(self):
        p=self.plan();p['goal_contract_ref']='../outside.json'
        result=prepare(p,ROOT)
        self.assertIn('goal_contract_ref:file_missing_or_outside_repository',result['blockers'])
        self.assertFalse(result['formal_experiment_ready'])

    def test_existing_engineering_goal_is_hashed_but_not_production_acceptance(self):
        p=self.plan();p['goal_contract_ref']='configs/goals/atropine_goal_regression.json'
        result=prepare(p,ROOT)
        self.assertEqual(len(result['reference_sha256']['goal_contract_ref']['sha256']),64)
        self.assertIn('engineering_regression_goal_not_production_acceptance',result['blockers'])

    def test_duplicate_seeds_and_implicit_order_are_rejected(self):
        p=self.plan();p['conditions'][0]['pair_seeds'][1]=p['conditions'][0]['pair_seeds'][0]
        with self.assertRaises(ValueError):prepare(p,ROOT)
        p=self.plan();p['run_order_seed']=None
        with self.assertRaises(ValueError):prepare(p,ROOT)

if __name__=='__main__':unittest.main()
