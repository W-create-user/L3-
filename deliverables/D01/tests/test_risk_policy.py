import unittest
from dataclasses import replace
from l3_system.monitoring.risk_policy import RiskEvidence,RiskPolicy

class RiskPolicyTests(unittest.TestCase):
    def evidence(self,v=0,**kw):
        return replace(RiskEvidence('e1',v,True,'valid',True,False,'in_scope'),**kw)

    def recover(self,p):
        p.evaluate(self.evidence(0));p.acknowledge_review('e1')
        p.evaluate(self.evidence(1));return p.evaluate(self.evidence(2))

    def test_ack_alone_never_recovers(self):
        p=RiskPolicy();p.evaluate(self.evidence());p.acknowledge_review('e1')
        self.assertEqual(p.evaluate(self.evidence(1))['risk_level'],2)

    def test_clean_alone_never_recovers(self):
        p=RiskPolicy()
        for v in range(5):self.assertEqual(p.evaluate(self.evidence(v))['risk_level'],2)

    def test_recovery_still_requires_gate(self):
        result=self.recover(RiskPolicy())
        self.assertEqual(result['risk_level'],1)
        self.assertFalse(result['execution_authorized']);self.assertTrue(result['requires_action_gate'])
        self.assertIsNone(result['confidence_probability'])

    def test_all_critical_evidence_blocks(self):
        for fields in ({'hard_rule_approved':False},{'context_current':False},{'data_quality':'unavailable'},
                       {'anomaly':True},{'anomaly':None},{'distribution':'unknown'},{'distribution':'out_of_scope'}):
            p=RiskPolicy();self.recover(p)
            result=p.evaluate(self.evidence(3,**fields))
            self.assertEqual(result['risk_level'],3);self.assertTrue(result['automatic_execution_blocked'])
            self.assertFalse(result['review_acknowledged'])

    def test_degraded_blocks(self):
        p=RiskPolicy();self.recover(p)
        self.assertTrue(p.evaluate(self.evidence(3,data_quality='degraded'))['automatic_execution_blocked'])

    def test_replay_revokes_recovery(self):
        p=RiskPolicy();self.recover(p)
        self.assertEqual(p.evaluate(self.evidence(2))['risk_level'],3)
        self.assertEqual(p.evaluate(self.evidence(3))['risk_level'],2)

    def test_gap_not_counted_consecutive(self):
        p=RiskPolicy();p.evaluate(self.evidence());p.acknowledge_review('e1')
        self.assertEqual(p.evaluate(self.evidence(5))['clean_observations'],1)

    def test_new_episode_requires_reset_and_review(self):
        p=RiskPolicy();self.recover(p)
        with self.assertRaises(ValueError):p.evaluate(self.evidence(0,episode_id='e2'))
        p.reset('e2')
        self.assertEqual(p.evaluate(self.evidence(0,episode_id='e2'))['risk_level'],2)
        with self.assertRaises(ValueError):p.acknowledge_review('e1')

    def test_bad_types_rejected(self):
        for fields in ({'state_version':True},{'anomaly':1},{'data_quality':'normal'},{'hard_rule_approved':1}):
            with self.assertRaises(ValueError):RiskPolicy().evaluate(self.evidence(**fields))
        with self.assertRaises(ValueError):RiskPolicy(1)

if __name__=='__main__':unittest.main()
