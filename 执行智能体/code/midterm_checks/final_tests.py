import json
from pathlib import Path
import sys
import unittest


def main():
    project=Path(__file__).resolve().parents[1]
    output=project/'midterm_verification/final_checks'
    output.mkdir(parents=True,exist_ok=True)
    a_root=project/'integration_abd_1103/member_A_11_02'
    sys.path.insert(0,str(a_root))
    modules=[
        'integration_abd_1103.test_contract','integration_abd_1103.test_peer_interfaces',
        'integration_abd_1103.test_current_task','integration_abd_1103.test_D_PR1',
        'data_collection_1105.test_termination','data_collection_1104.test_recorder',
    ]
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.loadTestsFromNames(modules),loader.discover(str(a_root/'tests'),pattern='test*.py')])
    with (output/'tests.txt').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    summary={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),
             'scope':'A original tests plus B/D contract, termination mapping and experiment recorder; unit stand-ins are not model experiments.'}
    (output/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
