import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_execution_logs import audit


class ArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'synthetic.jsonl'
        self.rows=[{'event':'environment_reset','seed':1,'observation':[0,0],'state_version':1},
                   {'event':'action_prepared','action_request':{'request_id':'R','task_id':'T'}},
                   {'event':'execution_result','result':{'request_id':'R','task_id':'T','executed':False,'execution_status':'rejected'}}]

    def result(self):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        return audit(self.path)

    def test_rejected_candidate_never_counted_as_execution(self):
        report=self.result();self.assertEqual(report['executed_calls_reported'],0)
        self.assertEqual(report['rejected_candidates'],1);self.assertFalse(report['training_ready'])
        self.assertEqual(report['consistency_errors'],[])

    def test_duplicate_result(self):
        self.rows.append(self.rows[-1]);self.assertTrue(self.result()['consistency_errors'])

    def test_missing_request(self):
        self.rows.pop(1);self.assertTrue(self.result()['consistency_errors'])

    def test_wrong_task(self):
        self.rows[-1]['result']['task_id']='other';self.assertTrue(self.result()['consistency_errors'])

    def test_rejected_cannot_be_executed(self):
        self.rows[-1]['result']['executed']=True;self.assertTrue(self.result()['consistency_errors'])

    def test_multiple_calls_are_one_reset_record(self):
        self.rows[-1]['result'].update(executed=True,execution_status='completed')
        self.rows.extend([{'event':'action_prepared','action_request':{'request_id':'R2','task_id':'T2'}},
                          {'event':'execution_result','result':{'request_id':'R2','task_id':'T2','executed':True,'execution_status':'completed'}}])
        report=self.result();self.assertEqual(report['reset_records'],1)
        self.assertEqual(report['executed_calls_reported'],2)
        self.assertFalse(report['training_ready'])

    def test_malformed_request_identity_does_not_crash(self):
        self.rows[-1]['result']['request_id']=[]
        self.assertTrue(self.result()['consistency_errors'])

    def test_reset_cannot_erase_pending_request(self):
        self.rows.pop()
        self.rows.append({'event':'environment_reset','seed':2,'observation':[0,0],'state_version':2})
        self.assertTrue(self.result()['consistency_errors'])


if __name__=='__main__':unittest.main()
