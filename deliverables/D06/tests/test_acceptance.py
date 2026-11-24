from pathlib import Path
import tempfile
import unittest
from l3_system.monitoring.takeover_acceptance import run

class TakeoverAcceptanceTests(unittest.TestCase):
    def test_http_and_session_failure_paths_preserve_action_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=run(Path(tmp)/'acceptance')
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['cases']),7)
            by={r['case']:r for r in report['cases']}
            self.assertEqual(by['fixture_unknown_result']['total_step_calls'],0)
            self.assertEqual(by['fixture_unknown_result']['state'],'takeover_requested')
            for r in report['cases']:
                self.assertFalse(r['human_factor_study'])
                for e in r['events']:
                    if e['event']=='http_response' and e['route']=='/resume':
                        self.assertEqual(e['step_delta'],0)

if __name__=='__main__':unittest.main()
