"""November 4: collect real A/D/B/SMPL experiments, with complete version metadata."""
import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent
from integration_abd_1103.a_integration import load_decision_agent, run_a_plan
from integration_abd_1103.demo_actual_a import load_monitor

from .provenance import runtime_manifest, sha256
from .recorder import ExperimentRecorder, timestamp, write_json
from .validate_data import validate_run


def state_snapshot(adapter):
    try:
        observation = adapter.get_observation()
    except RuntimeError:
        observation = None
    return {'observation': observation, 'step_count': adapter.step_count,
            'step_calls': adapter.step_calls, 'state_version': adapter.state_version,
            'done': adapter.done}


def internal_initial_state(adapter, state, seed):
    env = adapter._env
    def vector(value):
        if hasattr(value, 'full'):
            value = value.full()
        return np.asarray(value).reshape(-1).tolist()
    return {'seed': seed, 'reset_method': 'original SMPL reset with NumPy global seed',
            'state': copy.deepcopy(state),
            'simulator_internal_snapshot': True,
            'internal_xk': vector(env.xk), 'internal_zk': vector(env.zk),
            'internal_estimator_xe': vector(env.xe),
            'source_config_snapshot': '../../inputs/atropineenv',
            'replay_note': 'Recreate the recorded model/options and reset with the seed; do not pass a 39-value observation as SMPL initial_state.'}


def collect_episode(output, episode_id, *, steps, include_rejection, seed,
                    adapter, monitor, decision, run_id, dt):
    directory = output / 'episodes' / episode_id
    with ExperimentRecorder(directory, run_id=run_id, episode_id=episode_id,
                            observation_dim=int(np.prod(adapter.observation_space.shape)),
                            action_dim=int(np.prod(adapter.action_space.shape)), dt_minutes=dt) as recorder:
        executor = ExecutionAgent(adapter, directory / 'B_events.jsonl')
        monitor.clear_logs()
        executor.reset(seed=seed)
        initial = state_snapshot(adapter)
        recorder.record_initial(initial, internal_initial_state(adapter, initial, seed))
        write_json(directory / 'episode_manifest.json', {
            'episode_id': episode_id, 'seed': seed, 'requested_steps': steps,
            'include_rejection': include_rejection, 'reset_count': 1,
            'action_policy': 'B local space_midpoint interface baseline',
            'note': 'One shared environment instance is advanced continuously; rejected candidates never advance it.'})
        schedule = ['normal'] * steps
        if include_rejection:
            schedule.insert(min(2, steps), 'reject')
        action_sequence = []
        stop_reason = 'requested_steps_collected'
        for kind in schedule:
            attempt = recorder.attempts + 1
            main_file = f'main_attempt_{attempt:04d}.jsonl'
            goal = {'goal_type': 'control_cycle', 'description': '11月4日数据采集控制周期',
                    'priority': 3, 'constraints': {'environment': 'AtropineEnv', 'require_monitoring': True}}
            before = state_snapshot(adapter)
            started = time.perf_counter()
            try:
                action = None
                if kind == 'reject':
                    action = adapter.action_space.high + (adapter.action_space.high - adapter.action_space.low)
                trace = run_a_plan(decision, goal, executor, monitor, explicit_action=action,
                                   log_path=directory / main_file)
                duration = (time.perf_counter() - started) * 1000
                after = state_snapshot(adapter)
                result = trace['execution_result']
                if kind == 'reject' and (result['executed'] or result['success'] or result['execution_status'] != 'rejected'):
                    raise AssertionError('The deliberately out-of-range candidate was not blocked')
                recorder.record_attempt(trace, before, after,
                                        wall_duration_ms=duration, main_trace_file=main_file)
                action_sequence.append({
                    'attempt_index': attempt, 'test_input_kind': kind,
                    'task_id': result['task_id'], 'request_id': result['request_id'],
                    'action': trace['action_request']['action'],
                    'action_source': trace['action_request']['action_source'],
                    'approved': result['safety_decision']['approved'],
                    'executed': result['executed'], 'reward': result['reward'],
                    'done': result['done'], 'step_count_before': before['step_count'],
                    'step_count_after': after['step_count']})
                print(f"{episode_id} attempt={attempt} kind={kind} status={result['execution_status']} step={after['step_count']} reward={result['reward']}", flush=True)
                if kind == 'reject':
                    continue
                if not result['success']:
                    stop_reason = 'environment_or_execution_failed'
                    break
                if result['done']:
                    stop_reason = 'environment_terminated'
                    break
            except Exception as error:
                recorder.record_exception(error, before=before, after=state_snapshot(adapter),
                                          main_trace_file=main_file, traceback_text=traceback.format_exc())
                stop_reason = 'exception'
                print(f'{episode_id}: stopped after exception: {error}', flush=True)
                break
        write_json(directory / 'action_sequence.json', action_sequence)
        monitor.export_logs(directory / 'D_raw_reviews.jsonl')
        summary = recorder.finish(requested_steps=steps, stop_reason=stop_reason)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--monitor-file')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    if not 1 <= args.steps <= 60 or not 0 <= args.seed < 2**32:
        parser.error('steps must be 1..60 and seed must be 0..2**32-1')
    project = Path(__file__).resolve().parents[1]
    monitor_file = Path(args.monitor_file or project / 'external_D/pr1_8c105883/source/agents/monitoring_agent.py').resolve()
    run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    output = Path(args.output_root or project / 'data_1104/runs') / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 'BExperiment/1.0', 'run_id': run_id, 'created_at': timestamp(),
                'status': 'setting_up', 'episode_ids': ['normal_baseline','with_rejection'],
                'requested_steps_per_episode': args.steps, 'seed': args.seed}
    write_json(output / 'manifest.json', manifest)
    try:
        decision, a_root = load_decision_agent()
        with EnvironmentAdapter() as adapter:
            monitor, provenance = load_monitor(adapter.action_space, monitor_file)
            if provenance != 'D_provided_source' or monitor.demo_mode:
                raise RuntimeError('Real D source and actual environment bounds are required for this experiment')
            env = adapter._env
            from smpl.envs.models.config import dt
            manifest.update(runtime_manifest(adapter.source_dir, a_root, monitor_file, project))
            manifest['actual_D_code_used'] = True
            manifest['actual_SMPL_used'] = True
            manifest['runtime_options'] = {name: getattr(env, name) for name in [
                'normalize','uss_subtracted','dense_reward','max_steps','error_reward',
                'reward_scaler','reward_on_steady','reward_on_ess_subtracted',
                'reward_on_absolute_efactor','reward_on_actions_penalty','reward_on_reject_actions',
                'relaxed_max_min_actions','observation_include_t','observation_include_action',
                'observation_include_uss','observation_include_ess','observation_include_e',
                'observation_include_kf','observation_include_z','observation_include_x']}
            manifest['sampling_interval_minutes'] = dt
            manifest['observation_space'] = {'shape':list(adapter.observation_space.shape),
                'dtype':str(adapter.observation_space.dtype), 'low':adapter.observation_space.low.tolist(),
                'high':adapter.observation_space.high.tolist()}
            manifest['action_space'] = {'shape':list(adapter.action_space.shape),
                'dtype':str(adapter.action_space.dtype), 'low':adapter.action_space.low.tolist(),
                'high':adapter.action_space.high.tolist()}
            manifest['status'] = 'running'
            inputs = output / 'inputs/atropineenv'
            inputs.mkdir(parents=True)
            for filename in ['x0.txt','z0.txt','model.npy']:
                shutil.copy2(Path(adapter.source_dir) / 'smpl/configdata/atropineenv' / filename, inputs / filename)
            manifest['input_snapshot_sha256'] = {str(p.relative_to(output)):sha256(p) for p in inputs.iterdir()}
            write_json(output / 'manifest.json', manifest)
            summaries = []
            for name, rejection in [('normal_baseline',False),('with_rejection',True)]:
                summaries.append(collect_episode(output, name, steps=args.steps, include_rejection=rejection,
                                                 seed=args.seed, adapter=adapter, monitor=monitor,
                                                 decision=decision, run_id=run_id, dt=dt))
        validation = validate_run(output)
        completed = all(summary['experiment_status'] == 'completed' for summary in summaries)
        verification = {'status': 'PASS' if completed else 'EXPERIMENT_STOPPED_EARLY',
                        'data_integrity': validation, 'episodes': summaries,
                        'meaning': 'Data collection functionality verified; no production performance improvement is claimed'}
        write_json(output / 'verification.json', verification)
        with (output / 'experiment_index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        manifest['status'] = 'completed' if completed else 'stopped_early'
        manifest['finished_at'] = timestamp()
        write_json(output / 'manifest.json', manifest)
        print(json.dumps(verification, ensure_ascii=False), flush=True)
        print('Evidence:', output, flush=True)
        if not completed:
            raise SystemExit(2)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['failure_reason'] = str(error)
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'fatal_error.json', {'error_type':type(error).__name__, 'reason':str(error),
                                                'traceback':traceback.format_exc()})
        print('Failed evidence:', output, flush=True)
        raise


if __name__ == '__main__':
    main()
