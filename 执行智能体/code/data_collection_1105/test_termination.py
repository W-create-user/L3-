import unittest
from integration_abd_1103.environment_adapter import legacy_termination_flags


class TerminationMappingTests(unittest.TestCase):
    def test_continuing(self):
        self.assertEqual(legacy_termination_flags(False,{'terminal':False,'timeout':False}),(False,False))

    def test_timeout_is_truncation_not_environment_termination(self):
        self.assertEqual(legacy_termination_flags(True,{'terminal':True,'timeout':True}),(False,True))

    def test_environment_error_is_termination(self):
        self.assertEqual(legacy_termination_flags(True,{'terminal':True,'timeout':False}),(True,False))

    def test_legacy_fallback_without_timeout(self):
        self.assertEqual(legacy_termination_flags(True,{}),(True,False))

    def test_no_done_does_not_create_a_terminal_flag(self):
        self.assertEqual(legacy_termination_flags(False,{}),(False,False))


if __name__=='__main__':unittest.main()
