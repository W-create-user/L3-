from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from l3_system.monitoring.event_metrics import evaluate_events,aggregate_events


class EventMetricTests(unittest.TestCase):
    def test_later_miss_not_hidden_by_first_detection(self):
        report=evaluate_events([False,True,True,False,False,True,True,False],[True,False,False,False,False,False,False],2)
        self.assertEqual(report['event_count'],2);self.assertEqual(report['detected_event_count'],1)
        self.assertEqual(report['missed_event_count'],1);self.assertIsNone(report['events'][1]['delay_records'])

    def test_delay_measured_from_raw_onset(self):
        report=evaluate_events([False,False,True,True,True,False],[False,False,True,False],3)
        self.assertEqual(report['events'][0]['first_detection_record'],4)
        self.assertEqual(report['events'][0]['delay_records'],2)

    def test_post_event_window_tail_is_not_detection(self):
        report=evaluate_events([False,False,True,False,False,False],[False,True,False,False],3)
        self.assertEqual(report['detected_event_count'],0);self.assertEqual(report['missed_event_count'],1)
        self.assertEqual(report['false_alarm_window_count'],0)  # tail window truth still anomalous

    def test_dilated_windows_do_not_merge_distinct_raw_events(self):
        report=evaluate_events([False,False,True,False,True,False],[True,False,True,False],3)
        self.assertEqual(report['event_count'],2);self.assertEqual(report['detected_event_count'],2)

    def test_warmup_unscorable_not_silent_miss_or_zero_delay(self):
        report=evaluate_events([True,True,False,False,False],[True,False],4)
        self.assertEqual(report['unscorable_event_count'],1)
        self.assertEqual(report['missed_event_count'],0);self.assertIsNone(report['scorable_event_recall'])

    def test_event_spanning_warmup_keeps_true_onset(self):
        report=evaluate_events([True,True,True,True,False],[True,False],4)
        self.assertEqual(report['events'][0]['delay_records'],3)
        self.assertTrue(report['events'][0]['onset_before_first_window'])

    def test_normal_episode_false_alarm_separate_denominator(self):
        report=evaluate_events([False]*5,[False,True,True,False],2)
        self.assertTrue(report['all_normal_episode_with_alarm']);self.assertEqual(report['false_alarm_window_count'],2)
        self.assertIsNone(report['scorable_event_recall'])

    def test_no_alarms_keeps_miss(self):
        report=evaluate_events([False,True,True,False],[False]*3,2)
        self.assertEqual(report['missed_event_count'],1)
        self.assertIsNone(aggregate_events([report])['mean_delay_records_among_detected'])

    def test_aggregate_preserves_all_denominators(self):
        one=evaluate_events([False,True,True,False],[True,False,False],2)
        two=evaluate_events([False]*4,[False,True,False],2)
        total=aggregate_events([one,two])
        self.assertEqual(total['episode_count'],2);self.assertEqual(total['window_count'],6)
        self.assertEqual(total['all_normal_episode_count'],1);self.assertEqual(total['event_count'],1)
        self.assertEqual(total['detected_events_with_delay'],1)

    def test_numeric_truth_and_wrong_lengths_rejected(self):
        with self.assertRaises(ValueError):evaluate_events([0,1,1],[True,False],2)
        with self.assertRaises(ValueError):evaluate_events([False]*4,[False],2)
        with self.assertRaises(ValueError):evaluate_events([True]*4,[False]*4,1)

    def test_empty_aggregate_is_explicit(self):
        total=aggregate_events([])
        self.assertEqual(total['episode_count'],0);self.assertIsNone(total['scorable_event_recall'])

    def test_last_sample_event_evaluated(self):
        report=evaluate_events([False,False,False,True],[False,False,True],2)
        self.assertEqual(report['events'][0]['delay_records'],0)


if __name__=='__main__':unittest.main()
