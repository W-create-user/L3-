"""Export actual observation semantics/examples and directly exercise D's old API."""
import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import EnvironmentAdapter, ExecutionAgent
from .demo_actual_a import load_monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-file', required=True)
    args = parser.parse_args()
    output = Path(__file__).parent / 'exports_D' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    checks = []

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append({'check': name, 'status': 'PASS'})
        print('PASS:', name, flush=True)

    with EnvironmentAdapter() as adapter:
        monitor, provenance = load_monitor(adapter.action_space, args.monitor_file)
        assert provenance == 'D_provided_source'
        env = adapter._env  # Introspection only: never mutate the upstream model.
        flags = {name: getattr(env, name) for name in [
            'observation_include_t', 'observation_include_action', 'observation_include_uss',
            'observation_include_ess', 'observation_include_e', 'observation_include_kf',
            'observation_include_z', 'observation_include_x']}
        expected = dict(zip(flags, [True, False, True, True, True, True, True, False]))
        check('Observation layout matches the verified upstream configuration', flags == expected and adapter.observation_space.shape == (39,))
        executor = ExecutionAgent(adapter, output / 'B_api_events.jsonl')
        initial = executor.reset(seed=1101)
        reset_dtype = str(adapter._observation.dtype)
        task = {'task_id': 'D-REAL-API-001', 'task_type': 'control_cycle', 'description': 'D two-argument API verification'}
        request = executor.prepare_action(task)
        before = adapter.step_calls
        raw = monitor.check_action(initial, request['action'])
        save('check_action_raw_pass.json', raw)
        check('D two-argument check_action approves without stepping', raw['approved'] is True and adapter.step_calls == before)
        result = executor.execute_action(request, raw)
        check('B directly accepts D raw approved response and steps once', result['success'] and adapter.step_calls == before + 1)
        after_step = executor.get_observation()
        step_dtype = str(adapter._observation.dtype)
        save('ActionRequest.json', request)
        save('ExecutionResult.json', result)

        invalid = adapter.action_space.high + (adapter.action_space.high - adapter.action_space.low)
        blocked_request = executor.prepare_action({'task_id': 'D-REAL-API-002'}, action=invalid)
        before_reject = adapter.step_calls
        raw_reject = monitor.check_request(blocked_request, executor.get_observation())
        save('check_request_raw_reject.json', raw_reject)
        blocked = executor.reject_action_request(blocked_request, raw_reject)
        save('RejectedResult.json', blocked)
        check('D check_request rejection returns R-02 without stepping',
              raw_reject['rule_id'] == 'R-02' and adapter.step_calls == before_reject)

        invalid_inputs = [('missing', None), ('empty', []), ('wrong_shape', [0]),
                          ('not_numeric', ['bad'] * 4),
                          ('NaN', np.full(adapter.action_space.shape, np.nan)),
                          ('Inf', np.full(adapter.action_space.shape, np.inf))]
        invalid_results = []
        for name, value in invalid_inputs:
            count = adapter.step_calls
            answer = monitor.check_action(after_step, value, task_id='D-R01-' + name)
            check('Actual D rejects ' + name, answer['approved'] is False
                  and answer['rule_id'] == 'R-01' and adapter.step_calls == count)
            invalid_results.append({'case': name, 'input_repr': repr(value),
                                    'decision': answer, 'step_delta': adapter.step_calls - count})
        save('R01_invalid_cases.json', invalid_results)
        check('Real action bounds disable D demo mode', monitor.demo_mode is False)

        from smpl.envs.helpers.constants import COMPONENTS, USS
        from smpl.envs.models.config import dt
        fields = []
        def field(index, name, meaning, unit, group):
            fields.append({'index': index, 'upstream_name': env.observation_name[index],
                           'name': name, 'meaning': meaning, 'unit': unit, 'group': group,
                           'model_box_low': float(adapter.observation_space.low[index]),
                           'model_box_high': float(adapter.observation_space.high[index]),
                           'reset_example': initial[index], 'step_example': after_step[index]})
        field(0, 'step_index_before_increment', '模型在 step_count 自增前写入的步号，非物理分钟数', 'step_index', 'time_index')
        for i in range(4):
            field(1+i, f'USS_{i+1}', '第 '+str(i+1)+' 路基准输入流量，不是当前候选偏移', 'mL/min (inferred from u/1000 conversion)', 'steady_inputs')
        field(5, 'E_factor_steady', '基准 E-factor，env.yss', 'dimensionless', 'efactor')
        field(6, 'E_factor_current', '当前 E-factor，env.previous_efactor', 'dimensionless', 'efactor')
        for i in range(2):
            field(7+i, f'estimator_x_{i+1}', '状态估计器坐标，未确认可直接对应的现场物理量', 'model_coordinate_unspecified', 'estimator')
        for phase, start in [('organic', 9), ('aqueous', 23)]:
            for i, component in enumerate(COMPONENTS):
                field(start+i, f'F_{phase}_{i}', phase+' 相 '+component+' 的摩尔流量（z 代数状态）', 'mol/min', 'separator_algebraic')
        field(37, 'Q_organic', '分离器有机相体积流量，z[28]', 'L/min', 'separator_algebraic')
        field(38, 'Q_aqueous', '分离器水相体积流量，z[29]', 'L/min', 'separator_algebraic')
        check('All 39 observation indices are covered exactly once', [f['index'] for f in fields] == list(range(39)))
        schema = {
            'format': 'flat list of 39 JSON numbers, not a named state dictionary',
            'transport': 'B get_observation returns a copied list[float]',
            'observation_space_dtype': str(adapter.observation_space.dtype),
            'upstream_reset_dtype': reset_dtype, 'upstream_step_dtype': step_dtype,
            'normalize': False, 'flags': flags, 'sampling_interval_minutes': dt,
            'step_count_after_normal_execution': adapter.step_count,
            'timing_caution': 'Upstream appends step_count before increment: reset[0]=0 and first step[0]=0 while returned step_count=1. Use external step_count for sample alignment.',
            'clipping_caution': 'Upstream clips returned step observations to the model Box; do not infer all physical states were in range merely from returned values.',
            'threshold_caution': 'model_box_low/high are environment definitions, not certified plant safety thresholds. R-03 thresholds remain for D to define with documented sources.',
            'estimator_caution': 'Indices 7/8 can be negative; do not assume every observation must be nonnegative.',
            'source_files': ['smpl/envs/atropineenv.py', 'smpl/envs/models/atropine_process.py', 'smpl/envs/helpers/constants.py', 'smpl/envs/models/config.py'],
            'fields': fields,
        }
        save('observation_schema.json', schema)
        save('observation_samples.json', {'seed':1101, 'reset':initial, 'after_one_step':after_step,
                                         'step_count':adapter.step_count, 'state_version':adapter.state_version})
        save('action_space.json', {'shape':list(adapter.action_space.shape), 'dtype':str(adapter.action_space.dtype),
                                  'low':adapter.action_space.low.tolist(), 'high':adapter.action_space.high.tolist(),
                                  'normalize':False, 'uss_subtracted':True, 'USS':USS.tolist(),
                                  'normal_candidate':request['action'], 'rejected_candidate':blocked_request['action']})
        monitor.export_logs(output / 'D_actual_logs.jsonl')
        save('D_statistics.json', monitor.get_statistics())
        save('manifest.json', {'monitor_file':str(Path(args.monitor_file).resolve()),
                              'monitor_sha256':hashlib.sha256(Path(args.monitor_file).read_bytes()).hexdigest(),
                              'D_implementation_used':True, 'SMPL_source':adapter.source_dir,
                              'B_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                              'model_source_sha256':{p:hashlib.sha256((Path(adapter.source_dir)/p).read_bytes()).hexdigest() for p in schema['source_files']}})
        save('verification.json', {'status':'PASS','actual_D_code_used':True,'real_SMPL_used':True,
                                   'direct_check_action_step_delta':1,'rejected_step_delta':0,
                                   'R03_enabled':False,'checks':checks})
        print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
