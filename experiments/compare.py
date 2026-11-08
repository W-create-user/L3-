"""Paired Atropine baselines plus an explicitly offline safety-gate comparison."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from l3_system.execution import EnvironmentAdapter, ExecutionAgent
from l3_system.execution.local_policy import space_midpoint
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.algorithms.cnp_protocol import FixedOrderProtocol
from l3_system.main import run_control_cycle


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def rate(count, total):
    return count / total if total else None


def aggregate(rows):
    groups = {}
    for arm in sorted({r['arm'] for r in rows}):
        samples = [r for r in rows if r['arm'] == arm]
        groups[arm] = {
            'task_count':len(samples),
            'task_completion_rate':rate(sum(r['success'] for r in samples), len(samples)),
            'mean_task_seconds':float(np.mean([r['elapsed_seconds'] for r in samples])),
            'assignment_success_rate':rate(sum(r['awards'] for r in samples), sum(r['assignment_attempts'] for r in samples)),
            'total_environment_steps':sum(r['steps'] for r in samples),
        }
    return groups


def safety_comparison(adapter, out, seed):
    """Only D's gate is disabled for the offline arm; env.step is never called."""
    rng = np.random.default_rng(seed)
    low, high = adapter.action_space.low.astype(float), adapter.action_space.high.astype(float)
    monitor = MonitoringAgent(adapter.action_space)
    safe = [low, high] + [rng.uniform(low, high) for _ in range(18)]
    dangerous = []
    for i in range(20):
        candidate = (low + high)/2
        axis = i % len(low)
        candidate[axis] = (high[axis] + (high[axis]-low[axis])*rng.uniform(.01,.5)
                           if (i // len(low)) % 2 else low[axis] - (high[axis]-low[axis])*rng.uniform(.01,.5))
        dangerous.append(candidate)
    rows = []
    before = adapter.step_calls
    for index, (action, unsafe) in enumerate([(x, False) for x in safe]+[(x, True) for x in dangerous]):
        decision = monitor.check_request({'task_id':f'offline-{index}', 'action':action.tolist()}, adapter.get_observation())
        for enabled in (True, False):
            blocked = decision['decision'] == 'reject' if enabled else False
            rows.append({'index':index, 'monitor_enabled':enabled, 'unsafe_by_construction':unsafe,
                         'blocked':blocked, 'action':json.dumps(action.tolist()),
                         'rule_id':decision['rule_id'] if enabled else None})
    if adapter.step_calls != before:
        raise RuntimeError('Offline gate comparison advanced the environment')
    with (out/'safety_samples.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    monitor.export_logs(out/'offline_D_reviews.jsonl')
    report = {}
    for enabled in (True, False):
        samples = [r for r in rows if r['monitor_enabled'] == enabled]
        unsafe = [r for r in samples if r['unsafe_by_construction']]
        safe = [r for r in samples if not r['unsafe_by_construction']]
        report['enabled' if enabled else 'disabled'] = {
            'unsafe_count':len(unsafe), 'safe_count':len(safe),
            'dangerous_candidate_interception_rate':rate(sum(r['blocked'] for r in unsafe),len(unsafe)),
            'normal_candidate_false_interception_rate':rate(sum(r['blocked'] for r in safe),len(safe)),
        }
    return report


def run_comparisons(out, seeds):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, paired = [], []
    for seed in seeds:
        outcomes = {}
        initial_states = []
        for arm in ('single_executor', 'four_fixed', 'four_cnp'):
            folder = out/f'{arm}_{seed}';folder.mkdir()
            with EnvironmentAdapter() as adapter:
                executor = ExecutionAgent(adapter, folder/'B_events.jsonl')
                initial_states.append(executor.reset(seed=seed))
                started = time.perf_counter()
                if arm == 'single_executor':
                    # Same policy/environment; directly execute without A/C/D orchestration.
                    outcome = adapter.step(space_midpoint(executor.get_observation(), adapter.action_space))
                    attempts = awards = 0
                    result = outcome
                else:
                    monitor = MonitoringAgent(adapter.action_space)
                    scheduler = SchedulingAgent(FixedOrderProtocol()) if arm == 'four_fixed' else SchedulingAgent()
                    result = run_control_cycle(goal='执行一次控制任务', executor=executor,
                        monitor=monitor, scheduler=scheduler, event_path=folder/'events.jsonl')
                    if result['summary']['run_status'] != 'completed' or not result['summary']['all_C_locks_released']:
                        raise RuntimeError(f'{arm} failed at seed {seed}')
                    outcome = next(s['execution_result'] for s in result['stage_results'] if s['task_type']=='execute_approved_action')
                    attempts = len(result['c_tasks']); awards = sum(len(h) for h in result['assignments'])
                    monitor.export_logs(folder/'D_reviews.jsonl')
                elapsed = time.perf_counter() - started
                if not outcome['success'] or adapter.step_calls != 1:
                    raise RuntimeError(f'{arm} did not advance exactly one successful step')
                outcomes[arm] = outcome
                rows.append({'seed':seed,'arm':arm, 'success':bool(outcome['success']),
                             'elapsed_seconds':elapsed,'steps':adapter.step_calls,
                             'assignment_attempts':attempts,'awards':awards,'reward':outcome['reward']})
                write_json(folder/'result.json', result)
        equal_initial = all(np.array_equal(initial_states[0], x) for x in initial_states[1:])
        equal_outcomes = all(np.allclose(outcomes['single_executor']['observation'], x['observation'], rtol=1e-10, atol=1e-10)
                             and np.isclose(outcomes['single_executor']['reward'], x['reward'])
                             and np.array_equal(outcomes['single_executor']['action'], x['action'])
                             for x in outcomes.values())
        if not equal_initial or not equal_outcomes:
            raise RuntimeError(f'Paired baseline diverged at seed {seed}')
        paired.append({'seed':seed,'equal_initial_states':equal_initial,'equal_actions_rewards_observations':equal_outcomes})
    with EnvironmentAdapter() as adapter:
        adapter.reset(seed=seeds[0])
        safety = safety_comparison(adapter, out, seeds[0])
    with (out/'runs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root/'l3_system').rglob('*.py')) + [Path(__file__)]
    summary = {'schema_version':1,'environment':'real SMPL.AtropineEnvGym',
               'config':{'seeds':seeds,'cycles_per_arm':1,'policy':'space_midpoint','paired_arms':['single_executor','four_fixed','four_cnp']},
               'python':platform.python_version(),
               'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'process_results':aggregate(rows), 'paired_checks':paired,
               'safety_gate_offline':safety,
               'limits':['Task completion means interface cycle completion, not achievement of a process production target.',
                         'Only one compatible local worker bids per stage: fixed/CNP comparison is a degenerate baseline, not evidence of CNP superiority.',
                         'Single executor bypasses orchestration; time measures software overhead, not production efficiency.',
                         'Safety labels are constructed from action bounds; disabled gate is offline and never executes dangerous actions.',
                         'Small sample: no statistical significance, generalization or industrial safety claims.']}
    write_json(out/'summary.json',summary)
    plot_summary(summary, out/'comparison.png')
    return summary


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = summary['process_results']
    arms = ['single_executor','four_fixed','four_cnp']
    labels = ['Single B','Four / fixed','Four / CNP']
    fig, axes = plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    axes[0].bar(labels,[100*groups[a]['task_completion_rate'] for a in arms])
    axes[0].set(title='Interface completion (n=3 per arm)',ylabel='Percent',ylim=(0,110))
    axes[1].bar(labels,[1000*groups[a]['mean_task_seconds'] for a in arms])
    axes[1].set(title='Mean software cycle time',ylabel='Milliseconds (not process efficiency)')
    safety = summary['safety_gate_offline']
    axes[2].bar(['Gate on','Gate off'],[100*safety[k]['dangerous_candidate_interception_rate'] for k in ['enabled','disabled']])
    axes[2].set(title='Offline constructed out-of-bounds (n=20)',ylabel='Intercepted percent',ylim=(0,110))
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('outputs/comparison'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[1101,1102,1103])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    result = run_comparisons(args.output,args.seeds)
    print(json.dumps({k:result[k] for k in ('process_results','safety_gate_offline')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
