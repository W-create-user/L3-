import json
from pathlib import Path
import tempfile
import unittest
from integrated_system.system import EnvironmentAdapter, FourAgentSystem

class IntegrationTests(unittest.TestCase):
    def run_case(self, scenario='normal', checker=None, goal='执行一次仿真控制任务'):
        with tempfile.TemporaryDirectory() as tmp, EnvironmentAdapter() as env:
            system=FourAgentSystem(env,Path(tmp)/'run',checker=checker)
            result=system.run(goal=goal,scenario=scenario)
            return result,system.events

    def assert_blocked(self,checker):
        result,_=self.run_case(checker=checker)
        self.assertEqual(result['step_delta'],0)
        self.assertEqual(result['status'],'failed')
        self.assertTrue(result['all_C_locks_released'])

    def test_normal_all_six_dependencies_and_ids(self):
        result,events=self.run_case()
        self.assertEqual(result['status'],'completed');self.assertEqual(result['step_delta'],1)
        steps=result['task']['subtasks'];self.assertEqual(len(steps),6)
        self.assertTrue(all(s['status']=='completed' for s in steps))
        assignments=[e for e in events if e['event']=='assignment_received']
        self.assertEqual([e['assignment']['subtask_id'] for e in assignments],[s['subtask_id'] for s in steps])
        self.assertTrue(all(e['assignment']['task_id']==result['task_id'] for e in assignments))
        self.assertTrue(result['execution_result']['request_id'])
        self.assertFalse(result['D_demo_mode']);self.assertTrue(result['all_C_locks_released'])

    def test_reject_no_step_and_C_feedback(self):
        result,events=self.run_case('reject')
        self.assertEqual(result['step_delta'],0);self.assertEqual(result['failure_reason'],'safety_rejected')
        self.assertEqual(result['execution_result']['safety_decision']['rule_id'],'R-02')
        self.assertEqual(result['task']['subtasks'][-1]['status'],'pending')
        self.assertFalse([e for e in events if e['event']=='C_feedback'][-1]['feedback']['exclude_failed_agent'])

    def test_inspect_three_subtasks_no_step(self):
        result,_=self.run_case(goal='检查当前仿真状态')
        self.assertEqual(result['status'],'completed');self.assertEqual(result['step_delta'],0)
        self.assertEqual(len(result['task']['subtasks']),3)

    def test_missing_review(self):self.assert_blocked(lambda r,o:None)
    def test_unknown_review(self):self.assert_blocked(lambda r,o:{'task_id':r['task_id'],'decision':'unknown'})
    def test_wrong_task(self):self.assert_blocked(lambda r,o:{'task_id':'WRONG','decision':'pass'})
    def test_conflicting_review(self):self.assert_blocked(lambda r,o:{'task_id':r['task_id'],'decision':'pass','approved':False})
    def test_checker_exception(self):
        def broken(r,o):raise RuntimeError('injected checker failure')
        self.assert_blocked(broken)

if __name__=='__main__':unittest.main()
