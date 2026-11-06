"""Run the four-agent prototype from repository root."""
import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
import json


def main():
    route = argparse.ArgumentParser(add_help=False)
    route.add_argument('--engine', choices=['legacy', 'a-final'], default='legacy')
    selected, remaining = route.parse_known_args()
    if selected.engine == 'a-final':
        from a_integration.main import main as run_a
        return run_a(remaining)
    from integrated_system.system import EnvironmentAdapter, FourAgentSystem
    parser=argparse.ArgumentParser(epilog="New A package: --engine a-final --offline-preview; legacy remains the default")
    parser.add_argument('--scenario',choices=['normal','reject','inspect'],default='normal')
    parser.add_argument('--seed',type=int,default=1101)
    parser.add_argument('--output',type=Path,default=Path('outputs/four_agents'))
    args=parser.parse_args(remaining)
    output=args.output/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    with EnvironmentAdapter() as adapter:
        system=FourAgentSystem(adapter,output)
        goal='检查当前仿真状态' if args.scenario=='inspect' else '执行一次仿真控制任务'
        result=system.run(goal,seed=args.seed,scenario='reject' if args.scenario=='reject' else 'normal')
    print(json.dumps({k:v for k,v in result.items() if k not in {'task','execution_result'}},ensure_ascii=False,indent=2))
    print('Evidence: '+str(output.resolve()))
    expected = ('failed', 0, 'safety_rejected') if args.scenario == 'reject' else ('completed', 0 if args.scenario == 'inspect' else 1, None)
    actual = (result['status'], result['step_delta'], result['failure_reason'])
    if actual != expected or not result['all_C_locks_released']:
        raise SystemExit(1)
if __name__=='__main__':main()
