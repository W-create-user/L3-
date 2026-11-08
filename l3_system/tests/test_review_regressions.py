import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.algorithms.cnp_protocol import Bid, CNPError, CNPProtocol, FixedOrderProtocol
from l3_system.main import run_live
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from experiments.compare import aggregate


def task(sid, parent='P'):
    return {'task_id':parent,'subtask_id':sid,'task_type':'read','dependencies':[],'status':'pending'}


def bid(sid, cost=1, agent='B'):
    return {'task_id':'P','subtask_id':sid,'agent_id':agent,'estimated_cost':cost,'capability_match':1,'load':0}


class ReviewRegressions(unittest.TestCase):
    def test_duplicate_feedback_cannot_release_next_task_lock(self):
        scheduler = SchedulingAgent()
        first, second = task('S1'), task('S2')
        scheduler.schedule(first,[bid('S1')])
        feedback = {'task_id':'P','subtask_id':'S1','execution_status':'completed'}
        scheduler.update_feedback(feedback)
        scheduler.schedule(second,[bid('S2')])
        scheduler.update_feedback(feedback)
        self.assertEqual(scheduler.active_agent_tasks,{'B':'S2'})
        self.assertEqual(len(scheduler.feedback_log),1)

    def test_terminal_feedback_cannot_be_reversed(self):
        scheduler = SchedulingAgent();scheduler.schedule(task('S'),[bid('S')])
        scheduler.update_feedback({'task_id':'P','subtask_id':'S','execution_status':'completed'})
        with self.assertRaises(CNPError):
            scheduler.update_feedback({'task_id':'P','subtask_id':'S','execution_status':'failed','failure_reason':'late'})
        self.assertEqual(scheduler.get_task_status('P','S'),'completed')

    def test_assignment_queries_check_parent(self):
        scheduler = SchedulingAgent();scheduler.schedule(task('S'),[bid('S')])
        for method in (scheduler.get_assignment,scheduler.get_active_assignment,scheduler.get_assignment_history):
            with self.subTest(method=method.__name__),self.assertRaises(CNPError):
                method('wrong','S')

    def test_ready_tasks_do_not_expose_mutable_internal_state(self):
        scheduler = SchedulingAgent();scheduler.register_task(task('S'))
        scheduler.ready_tasks()[0]['dependencies'].append('injected')
        self.assertEqual(scheduler.tasks['S']['dependencies'],[])

    def test_nonfinite_or_boolean_bid_values_refused(self):
        for value in (float('nan'),float('inf'),-float('inf'),True):
            for field in ('estimated_cost','capability_match','load'):
                data = bid('S');data[field] = value
                with self.subTest(field=field,value=value),self.assertRaises(CNPError):
                    Bid.from_mapping(data)

    def test_null_identity_refused(self):
        for field in ('task_id','agent_id'):
            for value in (None,'', '  '):
                data=bid('S');data[field]=value
                with self.subTest(field=field,value=value),self.assertRaises(CNPError):
                    Bid.from_mapping(data)

    def test_invalid_weights_refused(self):
        for value in (float('nan'),float('inf'),True,-1):
            with self.subTest(value=value),self.assertRaises(CNPError):
                CNPProtocol({'capability':value,'cost':1,'load':1})
        with self.assertRaises(CNPError):
            CNPProtocol({})

    def test_fixed_order_and_cnp_are_distinct(self):
        data = task('S'); offers=[bid('S',100,'A'),bid('S',0,'B')]
        fixed=FixedOrderProtocol();cnp=CNPProtocol()
        self.assertEqual(fixed.award(data,fixed.evaluate(fixed.collect_bids(data,offers)))['winner_id'],'A')
        self.assertEqual(cnp.award(data,cnp.evaluate(cnp.collect_bids(data,offers)))['winner_id'],'B')

    def test_aggregate_does_not_invent_assignment_rate_for_single_baseline(self):
        summary=aggregate([{'arm':'single','success':True,'elapsed_seconds':.1,'steps':1,'awards':0,'assignment_attempts':0}])
        self.assertIsNone(summary['single']['assignment_success_rate'])

    def test_parent_dependency_requires_every_child_completed(self):
        scheduler = SchedulingAgent()
        scheduler.register_tasks([task('S1'),task('S2')])
        child = task('NEXT','Q');child['dependencies']=['P']
        scheduler.register_task(child)
        scheduler.update_feedback({'task_id':'P','subtask_id':'S1','execution_status':'completed'})
        self.assertNotIn('NEXT',{t['subtask_id'] for t in scheduler.ready_tasks()})
        scheduler.update_feedback({'task_id':'P','subtask_id':'S2','execution_status':'completed'})
        self.assertIn('NEXT',{t['subtask_id'] for t in scheduler.ready_tasks()})

    def test_live_aggregate_reports_incomplete_episode(self):
        adapter=FakeAdapter()
        original=adapter.step
        def end(action):
            result=original(action);adapter.done=True
            result['done']=True;result['terminated']=True
            return result
        adapter.step=end
        context=MagicMock();context.__enter__.return_value=adapter
        with tempfile.TemporaryDirectory() as tmp,patch('l3_system.main.EnvironmentAdapter',return_value=context):
            result=run_live(cycles=3,output_dir=Path(tmp)/'run')
        self.assertEqual(result['summary']['run_status'],'incomplete_or_failed')
        self.assertEqual(result['summary']['failure_reason'],'episode_ended_before_requested_cycles')
        self.assertEqual(result['summary']['finished_cycles'],1)
        self.assertEqual(result['cycles'][0]['summary']['run_status'],'completed')
