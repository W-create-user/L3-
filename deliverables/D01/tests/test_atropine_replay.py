import tempfile
from pathlib import Path
import unittest
from l3_system.monitoring.atropine_replay import ReadOnlyReplayAdapter,exercise_episode

class AtropineReplayTests(unittest.TestCase):
    def rows(self,fault=True):
        rows=[]
        for i in range(5):
            values=[0.]*39;values[5]=10.;values[6]=12. if fault and i>=3 else 10.
            rows.append({'episode_id':'fixture','step':i,'delivered_observation':values,'done':False})
        return rows
    def test_actual_actors_block_cycle_and_both_old_request_paths(self):
        rows=self.rows()
        with tempfile.TemporaryDirectory() as tmp:
            r=exercise_episode({'episode_id':'fixture','split':'validation'},rows,[-1.]*4,[1.]*4,Path(tmp)/'logs')
        self.assertTrue(r['takeover_flow_exercised']);self.assertEqual(r['first_archived_trigger_step'],3)
        self.assertTrue(r['new_cycle_blocked']);self.assertTrue(r['old_approval_stale_blocked'])
        self.assertTrue(r['cancelled_request_replay_blocked']);self.assertTrue(r['local_resources_and_pending_clear'])
        self.assertEqual(r['step_attempts'],0);self.assertEqual(r['new_simulator_steps'],0)
    def test_clean_archive_not_executed_as_if_it_were_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:r=exercise_episode({'episode_id':'fixture','split':'train'},self.rows(False),[-1.]*4,[1.]*4,Path(tmp)/'logs')
        self.assertFalse(r['takeover_flow_exercised']);self.assertEqual(r['candidate_trigger_records'],0)
    def test_reset_step_and_rewind_are_forbidden(self):
        adapter=ReadOnlyReplayAdapter('fixture',self.rows(),[-1.]*4,[1.]*4);adapter.select_record(1)
        with self.assertRaises(RuntimeError):adapter.step([0.]*4)
        with self.assertRaises(RuntimeError):adapter.reset()
        with self.assertRaises(ValueError):adapter.select_record(1)
        with self.assertRaises(ValueError):adapter.select_record(0)
        self.assertEqual(adapter.step_calls,0)
    def test_input_is_read_only_and_observation_is_copied(self):
        rows=self.rows();adapter=ReadOnlyReplayAdapter('fixture',rows,[-1.]*4,[1.]*4);adapter.select_record(1)
        obs=adapter.get_observation();obs[5]=999
        self.assertEqual(rows[1]['delivered_observation'][5],10.)

if __name__=='__main__':unittest.main()
