"""B02: preserve real mAb trajectories and reject failed solver returns."""
from __future__ import annotations
import argparse
import contextlib
import datetime
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
try:
    import resource
except ImportError:
    resource = None

ROOT = Path(__file__).resolve().parents[1]


def validate_step(env, observation, reward, before):
    import numpy as np
    if (np.asarray(observation).shape != env.observation_space.shape
            or not np.isfinite(observation).all()):
        raise RuntimeError('Invalid live observation shape or values')
    if not np.isfinite(reward) or float(reward) == float(env.error_reward):
        raise RuntimeError('Solver failure/error reward must not count as a successful step')
    if env.step_count != before + 1:
        raise RuntimeError('Environment did not advance exactly one step')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1113)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=20)
    p.add_argument('--initial-deviation', type=float, default=0.01)
    p.add_argument('--inject-solver-failure', action='store_true', help='Offline fault test only; replace simulator callback with an exception')
    args = p.parse_args(argv)
    if args.steps < 1 or args.max_steps < 1 or not 0 <= args.initial_deviation <= 0.1:
        p.error('steps/max-steps must be positive; initial-deviation must be in [0,0.1]')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(), 'success': False,
              'backend': 'smpl.envs.mabenv.MAbEnvGym',
              'code_base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'configuration': {'seed':args.seed,'requested_steps':args.steps,'max_steps':args.max_steps,
                                'normalize':False,'initial_state_deviation_ratio':args.initial_deviation},
              'fault_injection': 'offline_simulator_callback_exception' if args.inject_solver_failure else None,
              'process_disturbance': None, 'sensor_noise': None, 'process_anomaly_label': None,
              'sampling_period': {'value':60,'unit':'min','basis':'mabenv.py constructor docstring'},
              'action_scale': 'physical = model_action * uscale; normalize=False remains scaled',
              'stages': [], 'successful_steps': 0}
    def save():
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def record(data):
        with (args.output/'trajectory.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    def stage(name, fn):
        entry={'name':name};report['stages'].append(entry);start=time.perf_counter()
        try:
            value=fn();entry['success']=True;return value
        except Exception:
            entry.update(success=False,traceback=traceback.format_exc());raise
        finally:
            entry['wall_seconds']=time.perf_counter()-start
            entry['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
            save()
    try:
        sys.path.insert(0,str(ROOT/'vendor/smpl'))
        mpc=stage('import_mpctools',lambda:importlib.import_module('mpctools'))
        required=['DiscreteSimulator','getCasadiFunc','nmpc']
        if not all(hasattr(mpc,s) for s in required):
            raise RuntimeError('Incorrect mpctools package: expected CasADi APIs')
        import numpy as np
        import casadi
        report['dependencies']={'mpctools_version':mpc.__version__,'mpctools_module':mpc.__file__,'casadi':casadi.__version__,'numpy':np.__version__}
        mod=stage('import_environment',lambda:importlib.import_module('smpl.envs.mabenv'))
        report['model_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['vendor/smpl/smpl/envs/mabenv.py','vendor/smpl/smpl/envs/helpers/mab_helpers.py','vendor/smpl/SOURCE_VERSION.json']}
        np.random.seed(args.seed)
        env=stage('initialize',lambda:mod.MAbEnvGym(dataset_dir=str(ROOT/'vendor/smpl/smpl/configdata/mabenv'),normalize=False,max_steps=args.max_steps,initial_state_deviation_ratio=args.initial_deviation))
        obs=stage('reset',env.reset)
        if np.asarray(obs).shape != env.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError('Invalid reset result')
        report['spaces']={'observation':list(env.observation_space.shape),'action':list(env.action_space.shape)}
        record({'kind':'reset','seed':args.seed,'step_count':0,'sim_time_min':0,'observation':np.asarray(obs).tolist()})
        if args.inject_solver_failure:
            def failed_simulator(*unused):
                raise RuntimeError('B02 injected offline solver failure')
            env._simulation=failed_simulator
        for index in range(args.steps):
            before=env.step_count
            action=np.asarray(env.steady_actions,dtype=env.action_space.dtype)
            if not env.action_space.contains(action):
                raise RuntimeError('Steady action outside live action space')
            def checked_step():
                buffer=io.StringIO();started=time.perf_counter()
                try:
                    with contextlib.redirect_stdout(buffer):
                        observation,reward,done,info=env.step(action)
                finally:
                    with (args.output/'solver_stdout.log').open('a',encoding='utf-8') as f:
                        f.write(buffer.getvalue())
                raw=np.asarray(env.previous_observation)
                row={'kind':'step','step_count':env.step_count,'sim_time_min':env.step_count*env.dt_spl,
                     'requested_action':action.tolist(),'observation':np.asarray(observation).tolist(),
                     'raw_preclip_observation':raw.tolist() if np.isfinite(raw).all() else None,
                     'raw_preclip_nonfinite_count':int((~np.isfinite(raw)).sum()),
                     'clipped_component_count':int(np.count_nonzero(raw != np.asarray(observation))),
                     'reward':float(reward) if np.isfinite(reward) else None,'done':bool(done),'info':info,
                     'wall_seconds':time.perf_counter()-started,'solver_stdout':buffer.getvalue()}
                record(row)  # Preserve failed returns before raising.
                validate_step(env,observation,reward,before)
                return bool(done),info
            done,info=stage(f'step_{index+1}',checked_step)
            report['successful_steps']+=1
            report['last_done']=done;report['last_info']=info
            if done:
                report['stop_reason']='model_done';break
        else:
            report['stop_reason']='requested_step_budget_reached'
        report['success']=True
    except Exception:
        report['traceback']=traceback.format_exc()
        print(report['traceback'],file=sys.stderr)
    finally:
        save()
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
