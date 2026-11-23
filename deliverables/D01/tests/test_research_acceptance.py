from copy import deepcopy
import unittest
from l3_system.monitoring.research_acceptance import audit

class AcceptanceTests(unittest.TestCase):
    def evidence(self):
        ev=dict(event_count=1,detected_event_count=1,missed_event_count=0,unscorable_event_count=0)
        counts=dict(TP=1,FP=1,FN=0,TN=1)
        summary=dict(episode_count=1,event_count=1,detected_event_count=1,missed_event_count=0,unscorable_event_count=0,window_count=3,all_normal_episode_count=0)
        part=dict(episodes=[dict(episode_id='fixture',confusion=counts,windows=3,event_evaluation=ev)],counts=counts.copy(),event_summary=summary)
        original=dict(label_scope='sensor_fault',physical_process_labels_available=False,evaluation={s:deepcopy(part) for s in ('validation','test')})
        fusion=dict(schema_version='D04DerivedSensorStudy/1.0',independent_research_acceptance_passed=False,original_inputs_unchanged=True,
            runs=[dict(variants={str(i):dict(private_artifact_receipt=dict(validation_roundtrip_exact=True)) for i in range(21)})])
        return original,fusion
    def test_reproduction_does_not_become_research_acceptance(self):
        a,f=self.evidence();r=audit(a,deepcopy(a),f)
        self.assertFalse(r['all_research_requirements_passed']);self.assertIsNone(r['sensor_evaluation']['test']['whole_normal_episode_false_alarm_rate'])
    def test_mismatch_rejected(self):
        a,f=self.evidence();b=deepcopy(a);b['label_scope']='physical'
        with self.assertRaises(ValueError):audit(a,b,f)
    def test_denominator_fraud_rejected(self):
        a,f=self.evidence();a['evaluation']['test']['episodes'][0]['windows']=99
        with self.assertRaisesRegex(ValueError,'denominator'):audit(a,deepcopy(a),f)
    def test_failed_checkpoint_rejected(self):
        a,f=self.evidence();f['runs'][0]['variants']['0']['private_artifact_receipt']['validation_roundtrip_exact']=False
        with self.assertRaises(ValueError):audit(a,deepcopy(a),f)
