import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from l3_system.monitoring.runtime_compatibility import require_supported_runtime,SUPPORTED_A_RUNTIMES
from l3_system.monitoring.status import monitor_status

class RuntimeCompatibilityTests(unittest.TestCase):
    def test_unknown_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runtime.py';path.write_text('unreviewed')
            with self.assertRaises(ValueError):require_supported_runtime(path)
    def test_only_exact_reviewed_bytes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runtime.py';path.write_text('synthetic unit fixture')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict(SUPPORTED_A_RUNTIMES,{sha:'test-fixture-only'},clear=True):
                self.assertEqual(require_supported_runtime(path)['reviewed_A_commit'],'test-fixture-only')
                path.write_text('changed fixture')
                with self.assertRaises(ValueError):require_supported_runtime(path)
    def test_status_is_not_a_safety_decision(self):
        monitor=SimpleNamespace(process_rules=SimpleNamespace(enabled=True,version='fixture',environment='SYNTHETIC'),
            statistical_r03_enabled=True,detector=SimpleNamespace(label_scope='synthetic_fixture'))
        result=monitor_status(monitor)
        self.assertTrue(result['r03_enabled']);self.assertTrue(result['statistical_r03_enabled'])
        self.assertFalse(result['execution_authorized'])

if __name__=='__main__':unittest.main()
