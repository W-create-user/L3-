import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_sensor_benchmark import phase_summary,shadow

class DiagnosticTests(unittest.TestCase):
    def test_raw_event_and_tail_are_separate(self):
        labels=[False]*4+[True]*3+[False]*5
        result=phase_summary(labels,[False]*9,[.1]*9,4)
        counts={k:v['windows'] for k,v in result['phases'].items()}
        self.assertEqual(counts,dict(uncorrupted_reference=0,before_injection=1,injection=3,
            post_injection_window_tail=3,recovered_window=2))
        self.assertEqual(result['phases']['post_injection_window_tail']['confusion']['FN'],3)
        self.assertEqual(result['warmup_records_excluded'],3)
    def test_pre_event_alarms_are_false_positives(self):
        labels=[False]*4+[True]*3+[False]*5
        result=phase_summary(labels,[True]*9,[.5]*9,4)
        self.assertEqual(result['phases']['before_injection']['confusion']['FP'],1)
        self.assertEqual(result['phases']['recovered_window']['confusion']['FP'],2)
    def test_uncorrupted_reference_does_not_mean_physical_normal(self):
        result=phase_summary([False]*8,[True]*5,[.5]*5,4)
        self.assertEqual(result['phases']['uncorrupted_reference']['confusion']['FP'],5)
    def test_bad_lengths_and_nan_rejected(self):
        for predictions,scores in (([False],[.1]),([False]*5,[float('nan')]*5)):
            with self.assertRaises(ValueError):phase_summary([False]*8,predictions,scores,4)
    def test_shadow_never_authorizes_and_keeps_audit_views_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'sensor_benchmark';folder.mkdir()
            (folder/'episodes.json').write_text(json.dumps({'episodes':[{'split':'validation','raw_trajectory':'fixture.jsonl'}]}))
            raw=[0.]*39;raw[5]=13.;raw[6]=13.1;delivered=raw.copy();delivered[6]*=1.2
            rows=[dict(step=0),dict(step=1,environment_observation=raw,delivered_observation=delivered,sensor_fault_label=1)]
            (folder/'fixture.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result=shadow(root);counts=result['counts']['validation']
            self.assertEqual(counts['status_disagreements'],1)
            self.assertEqual(counts['environment_observation'],{'within_candidate_band':1})
            self.assertFalse(result['runtime_enabled']);self.assertIsNone(result['operational_decision'])

if __name__=='__main__':unittest.main()
