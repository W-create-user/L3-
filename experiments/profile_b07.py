"""B07 local-chain wall-clock profiling. Raw artifacts stay in --output.

Requires the separately supplied B companion configurations. Measurements are
observations on this host, not latency SLAs or physical safety certification.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = ('atropine_midpoint', 'mab_steady', 'mab_midpoint', 'mab_PI_step',
         'mab_PI_noise', 'mab_MPC_step')


def distribution(values):
    import numpy as np
    if not values:
        return {'count': 0, 'mean_s': None, 'p95_s': None, 'max_s': None}
    return {'count': len(values), 'mean_s': float(np.mean(values)),
            'p95_s': float(np.percentile(values, 95)), 'max_s': max(values)}


class Timings:
    def __init__(self):
        self.rows = []

    def call(self, phase, fn, *args, **kwargs):
        started = time.perf_counter()
        error = None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            self.rows.append({'phase': phase, 'wall_seconds': time.perf_counter()-started,
                              'exception': error})

    def wrap(self, obj, name, phase):
        original = getattr(obj, name)
        setattr(obj, name, lambda *args, **kwargs: self.call(phase, original, *args, **kwargs))


def run_case(case, output):
    import numpy as np
    from l3_system.execution import EnvironmentAdapter, MAbAdapter, ExecutionAgent
    from l3_system.agents.monitoring_agent import MonitoringAgent
    from l3_system.runtime import MultiAgentSystem
    from l3_system.execution.controllers import BufferLevelPID, SpaceMidpointPolicy
    from l3_system.execution.controllers.source_mpc import SourceMPCPolicy
    from l3_system.planning.goal_runtime import run_numeric_goal
    from .run_b04_control_study import make_goal
    timings = Timings()
    report = {'case': case, 'complete': False, 'research_only': True,
              'R03_runtime_enabled': False, 'physical_safety_certified': False}
    started = time.perf_counter()
    try:
        with (output/'stdout.log').open('w') as log, contextlib.redirect_stdout(log):
            backend = EnvironmentAdapter if case.startswith('atropine') else MAbAdapter
            with timings.call('construct', backend, max_steps=12) as adapter:
                kw = {'seed': 231123}
                if backend is MAbAdapter:
                    state = adapter._env.steady_observations.copy()
                    state[17] += .3 / adapter._xscale[17]
                    kw['initial_state'] = state
                timings.call('reset', adapter.reset, **kw)
                report['metadata'] = adapter.metadata
                report['initial_state_sha256'] = hashlib.sha256(
                    np.asarray(adapter.get_observation(), dtype=np.float64).tobytes()).hexdigest()
                step_records = []
                original_step = adapter.step
                def measured_step(action):
                    record = timings.call('step', original_step, action)
                    step_records.append(record)
                    with (output/'steps.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    return record
                adapter.step = measured_step
                executor = ExecutionAgent(adapter, log_path=output/'B_events.jsonl')
                monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
                timings.wrap(monitor, 'check_request', 'D_review')
                system = MultiAgentSystem(executor=executor, monitor=monitor, event_path=output/'events.jsonl')
                original_cycle = system.run_cycle
                def measured_cycle(*args, **kwargs):
                    before = adapter.step_calls
                    row_index = len(timings.rows)
                    try:
                        return timings.call('full_cycle', original_cycle, *args, **kwargs)
                    finally:
                        for row in timings.rows[row_index:]:
                            if row['phase'] == 'full_cycle':
                                row['step_delta'] = adapter.step_calls-before
                system.run_cycle = measured_cycle
                if backend is EnvironmentAdapter:
                    policy = SpaceMidpointPolicy()
                    for _ in range(5):
                        result = system.run_cycle(action_generator=lambda obs, space, attempt:
                            timings.call('policy_predict', policy.predict, obs, space))
                        with (output/'cycles.jsonl').open('a') as stream:
                            stream.write(json.dumps(result)+'\n')
                        if result['summary']['run_status'] != 'completed':
                            break
                    report['status'] = result['summary']['run_status']
                else:
                    config = json.loads((ROOT/'configs/controllers/mab_buffer_pi.json').read_text())
                    registry = json.loads((ROOT/'configs/environments/mab_physical_registry.json').read_text())
                    policy = (BufferLevelPID(adapter, config) if '_PI_' in case else
                              SourceMPCPolicy(adapter, horizon=2, allow_research=True) if '_MPC_' in case else
                              SpaceMidpointPolicy())
                    if case == 'mab_steady':
                        policy.predict = lambda *_: adapter._env.steady_actions.copy()
                    if case == 'mab_PI_noise':
                        original = policy.predict
                        rng = np.random.default_rng(231123)
                        def noisy(obs, space):
                            delivered = np.asarray(obs).copy()
                            noise_dm = float(rng.normal(0, .003))
                            delivered[17] += noise_dm / adapter._xscale[17]
                            with (output/'measurement_events.jsonl').open('a') as stream:
                                stream.write(json.dumps({'kind': 'synthetic_policy_sensor_noise', 'index': 17,
                                    'noise_dm': noise_dm, 'seed': 231123,
                                    'true_dm': float(obs[17]*adapter._xscale[17]),
                                    'delivered_dm': float(delivered[17]*adapter._xscale[17]),
                                    'D_and_goal_receive_true_state': True})+'\n')
                            return original(delivered, space)
                        policy.predict = noisy
                    timings.wrap(policy, 'reset', 'policy_reset')
                    timings.wrap(policy, 'predict', 'policy_predict')
                    result = run_numeric_goal(system, make_goal(config, registry, case), registry,
                        environment='MAbEnvGym', policy=policy, timeout_seconds=120)
                    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                    report.update(status=result['summary']['status'], reason=result['summary']['reason'],
                                  final_error_dm=result['samples'][-1]['observation'][17]-config['setpoint_dm'],
                                  solver_statuses=getattr(policy, 'solver_statuses', None),
                                  saturation_count=getattr(policy, 'saturation_count', None))
                successful_steps = sum(row.get('success') is True for row in step_records)
                report.update(successful_steps=successful_steps,
                              failed_step_returns=sum(row.get('success') is not True for row in step_records),
                              successful_simulation_time_min=successful_steps*adapter.metadata['sampling_period']['value'],
                              simulation_time_note='sum of successful step intervals; failed numerical calls are not credited',
                              expected_outcome_met=(report['status']=='completed' if case.startswith('atropine') else
                                  report['status']=='succeeded' if ('_PI_' in case or '_MPC_' in case) else
                                  report['status']=='failed'))
                report.update(complete=True, step_calls=adapter.step_calls,
                              all_C_locks_released=not bool(system.scheduler.active_agent_tasks))
    except Exception:
        report['traceback'] = traceback.format_exc()
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['peak_process_rss_KiB_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['timings'] = timings.rows
        report['phases'] = {phase: distribution([r['wall_seconds'] for r in timings.rows if r['phase']==phase])
                            for phase in sorted({r['phase'] for r in timings.rows})}
        # Counts of failed adapter returns differ from raised Python exceptions.
        report['executing_cycle_timing'] = distribution([row['wall_seconds'] for row in timings.rows
            if row['phase']=='full_cycle' and row.get('step_delta', 0)>0])
        report['timing_note'] = 'inclusive nested timings; D may review twice; read-only goal verification cycles excluded from executing_cycle_timing; small deterministic samples, descriptive p95 only'
        report['timing_exceptions'] = sum(r['exception'] is not None for r in timings.rows)
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', choices=CASES)
    p.add_argument('--timeout', type=float, default=180)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error('timeout must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    if a.case:
        report = run_case(a.case, a.output)
        return 0 if report['complete'] and report.get('expected_outcome_met') else 1
    inputs = ['configs/controllers/mab_buffer_pi.json', 'configs/environments/mab_physical_registry.json',
              'configs/training/mab_action_partitions.draft.json', 'requirements-runtime-20261120.lock']
    missing = [path for path in inputs if not (ROOT/path).is_file()]
    if missing:
        (a.output/'blocked.json').write_text(json.dumps({'missing_private_companions': missing}, indent=2))
        return 2
    snapshot = a.output/'frozen_inputs'
    snapshot.mkdir()
    import shutil
    hashes = {}
    for path in inputs:
        target = snapshot/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, target)
        hashes[path] = hashlib.sha256(target.read_bytes()).hexdigest()
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for top in ['l3_system', 'experiments', 'vendor/smpl'] for path in sorted((ROOT/top).rglob('*.py'))}
    manifest = {'date': '2026-11-23', 'provisional_for_24th_freeze': True,
        'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'input_hashes': hashes, 'source_hashes': source_hashes, 'seed': 231123,
        'observed_dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
        'cases': list(CASES), 'child_timeout_s': a.timeout,
        'C05_confirmed': False, 'MADDPG_training_enabled': False,
        'scope': 'local A/B/C/D chain; no real network transport or D learned model integration',
        'cases_are_not_independent_statistical_samples': True, 'results': []}
    for case in CASES + ('gateway_withdrawal', 'worker_exit'):
        destination = a.output/case
        cmd = ([sys.executable, '-m', 'experiments.profile_b07', '--case', case, '--output', str(destination)]
               if case in CASES else [sys.executable, '-m', ('experiments.verify_b07_worker_exit' if case=='worker_exit' else 'experiments.verify_joint_gateway'), '--output', str(destination)])
        with (a.output/(case+'.process.log')).open('w') as log:
            try:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                row = {'case': case, 'returncode': result.returncode, 'timed_out': False}
            except subprocess.TimeoutExpired:
                row = {'case': case, 'returncode': None, 'timed_out': True}
        summary_path = destination/'summary.json'
        row['summary'] = json.loads(summary_path.read_text()) if summary_path.exists() else None
        manifest['results'].append(row)
        (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return 0 if all(r['returncode']==0 for r in manifest['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
