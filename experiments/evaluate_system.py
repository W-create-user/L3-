"""Reproducible synthetic interface benchmarks, NOT real process experiments."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter
import numpy as np
from runtime.cluster import AgentCluster
from runtime.dispatch import Dispatcher, Worker
from agents.monitoring_agent import MonitoringAgent
from tests.fixtures import TestAdapter


def main():
    output = Path('experiments/results/system')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
        for seed in range(30):
            for mode in ['single', 'four_agents']:
                env = TestAdapter()
                start = perf_counter()
                if mode == 'single':
                    env.reset(seed)
                    candidate = np.zeros(2)
                    approved = MonitoringAgent(env.action_space).check_action(env.get_observation(), candidate)['approved']
                    success = env.step(candidate)['success'] if approved else False
                else:
                    result = AgentCluster(env, log_path=Path(temp) / 'events.jsonl').run('执行一次控制任务', seed=seed)
                    success = result['success']
                    audit.append({'seed': seed, 'mode': mode, 'result': result})
                rows.append(dict(experiment='architecture', seed=seed, mode=mode,
                                 success=int(success), seconds=perf_counter()-start,
                                 unsafe_admitted='', normal_blocked=''))
            for mode in ['fixed', 'cnp']:
                workers = [Worker('unavailable', {'execute'}, 10, .9), Worker('healthy', {'execute'}, 1, .1)]
                start = perf_counter()
                result = Dispatcher(workers).execute([{'task_id': f't{seed}', 'subtask_id': f's{seed}', 'task_type': 'control'}],
                          {f's{seed}': 'execute'}, lambda worker, task: worker == 'healthy', strategy=mode)
                audit.append({'seed': seed, 'mode': mode, 'result': result})
                rows.append(dict(experiment='scheduling', seed=seed, mode=mode,
                                 success=int(result['success']), seconds=perf_counter()-start,
                                 unsafe_admitted='', normal_blocked=''))
            rng = np.random.default_rng(seed)
            for mode in ['monitor_on', 'monitor_off']:
                monitor = MonitoringAgent(TestAdapter().action_space)
                for dangerous in [False, True]:
                    rng = np.random.default_rng(seed)
                    candidate = rng.uniform(-.8, .8, 2) if not dangerous else np.array([1.2, 0.])
                    start = perf_counter()
                    approved = monitor.check_action([0., 0.], candidate)['approved'] if mode == 'monitor_on' else True
                    # Counterfactual only: bypass candidates are never submitted to a real env.
                    rows.append(dict(experiment='safety', seed=seed, mode=mode,
                        success=int(approved != dangerous), seconds=perf_counter()-start,
                        unsafe_admitted=int(approved) if dangerous else '',
                        normal_blocked=int(not approved) if not dangerous else ''))
    (output / 'events.jsonl').write_text(''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in audit))
    path = output / 'raw.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        key = row['experiment'] + '/' + row['mode']
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, group in groups.items():
        summary[key] = {'n': len(group), 'success_rate': sum(r['success'] for r in group)/len(group),
                        'mean_seconds': sum(r['seconds'] for r in group)/len(group)}
        for field in ['unsafe_admitted', 'normal_blocked']:
            values = [r[field] for r in group if r[field] != '']
            if values:
                summary[key][field + '_rate'] = sum(values)/len(values)
    manifest = {'environment': 'synthetic interface fixture; not SMPL', 'seeds': list(range(30)),
                'python': platform.python_version(), 'numpy': np.__version__,
                'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'summary': summary}
    (output / 'summary.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
