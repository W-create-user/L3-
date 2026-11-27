"""Profiling must retain failures and distinguish an empty measurement set."""
import unittest
from experiments.profile_b07 import Timings, distribution


class ProfilingTests(unittest.TestCase):
    def test_failed_call_is_recorded_and_propagated(self):
        timer = Timings()
        def fail():
            raise RuntimeError('injected measurement failure')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            timer.call('step', fail)
        self.assertEqual(len(timer.rows), 1)
        self.assertIn('RuntimeError', timer.rows[0]['exception'])
        self.assertGreaterEqual(timer.rows[0]['wall_seconds'], 0)

    def test_missing_measurements_are_not_zero_latency(self):
        self.assertIsNone(distribution([])['p95_s'])
        result = distribution([1., 2., 3.])
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['max_s'], 3.)

    def test_wrapped_call_returns_actual_value(self):
        class Subject:
            def calculate(self, n):
                return n + 1
        subject = Subject()
        timer = Timings()
        timer.wrap(subject, 'calculate', 'policy')
        self.assertEqual(subject.calculate(4), 5)
        self.assertIsNone(timer.rows[0]['exception'])
