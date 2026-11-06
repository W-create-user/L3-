"""Three fresh-environment repetitions per fixed scenario; actual dates are preserved."""
import csv
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from integration_abd_1103 import EnvironmentAdapter
from integration_abd_1103.demo_actual_a import load_monitor
from data_collection_1104.provenance import runtime_manifest, sha256
from data_collection_1104.recorder import write_json, timestamp
from data_collection_1105.collect_for_D import collect_episode, validate_episode


def read_arrays(folder):
    with (folder/'transitions.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    observations=np.asarray([[float(r[f'obs_{i:03d}']) for i in range(39)] for r in rows])
    actions=np.asarray([[float(r[f'action_{i:02d}']) for i in range(4)] for r in rows])
    rewards=np.asarray([float(r['reward']) for r in rows])
    return rows,observations,actions,rewards


def main():
    project=Path(__file__).resolve().parents[1]
    output=project/'midterm_verification/runs'/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True)
    monitor_file=project/'external_D/pr1_8c105883/source/agents/monitoring_agent.py'
    reference={};metrics=[];validations=[]
    metadata={'created_at':timestamp(),'planned_task_date':'2026-11-06',
        'actual_execution_date':datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d'),
        'note':'Executed early for the upcoming midterm; not dated as a future execution.',
        'seed':1101,'steps':100,'repetitions_per_scenario':3,
        'fresh_environment_per_repetition':True,'max_steps':-1,
        'technical_tolerance':{'atol':1e-6,'rtol':1e-7,'purpose':'numerical reproducibility only, not a physical safety threshold'},
        'runner_sha256':sha256(__file__)}
    for scenario in ['baseline_seed1101','legal_input_change']:
        for number in range(1,4):
            parent=output/scenario/f'repeat_{number:02d}'
            parent.mkdir(parents=True)
            with EnvironmentAdapter(max_steps=-1) as adapter:
                monitor,source=load_monitor(adapter.action_space,monitor_file)
                assert source=='D_provided_source' and not monitor.demo_mode
                from smpl.envs.models.config import dt
                if 'runtime' not in metadata:
                    metadata['runtime']=runtime_manifest(adapter.source_dir,project/'integration_abd_1103/member_A_11_02',monitor_file,project)
                    write_json(output/'manifest.json',metadata)
                summary,rows=collect_episode(parent,scenario,1101,100,adapter,monitor,dt)
            folder=parent/'episodes'/scenario
            validations.append(validate_episode(folder,len(rows)))
            if not summary['full_continuous_horizon']:
                raise RuntimeError(f'Incomplete fixed-condition repetition: {folder}')
            raw,obs,actions,rewards=read_arrays(folder)
            initial=json.loads((folder/'initial_conditions.json').read_text(encoding='utf-8'))
            initial_x=np.asarray(initial['internal_xk']);initial_z=np.asarray(initial['internal_zk'])
            if scenario not in reference:
                reference[scenario]=(obs,actions,rewards,initial_x,initial_z)
            ref_obs,ref_actions,ref_rewards,ref_x,ref_z=reference[scenario]
            same_initial=bool(np.array_equal(initial_x,ref_x) and np.array_equal(initial_z,ref_z))
            same_actions=bool(np.array_equal(actions,ref_actions))
            obs_delta=float(np.max(np.abs(obs-ref_obs)))
            reward_delta=float(np.max(np.abs(rewards-ref_rewards)))
            within_tolerance=bool(np.allclose(obs,ref_obs,atol=1e-6,rtol=1e-7)
                                  and np.allclose(rewards,ref_rewards,atol=1e-6,rtol=1e-7))
            if not (same_initial and same_actions and within_tolerance):
                raise AssertionError(f'Fixed-condition reproduction check failed: {folder}')
            metric={'scenario':scenario,'repeat':number,'seed':1101,'steps':len(raw),
                'same_initial_xk_zk':same_initial,'same_action_sequence':same_actions,
                'max_abs_observation_difference':obs_delta,'max_abs_reward_difference':reward_delta,
                'total_reward':float(rewards.sum()),'mean_reward':float(rewards.mean()),
                'final_E_factor':float(obs[-1,6]),'max_E_factor':float(obs[:,6].max()),
                'technical_reproducibility_pass':within_tolerance,
                'relative_episode_path':str(folder.relative_to(output)).replace('\\','/')}
            metrics.append(metric)
            print(f'REPEAT PASS: {scenario} #{number} obs_delta={obs_delta} reward_delta={reward_delta}',flush=True)
    with (output/'reproducibility_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    result={'status':'PASS','actual_execution_date':metadata['actual_execution_date'],
        'scenarios':2,'repetitions':6,'total_real_steps':sum(m['steps'] for m in metrics),
        'maximum_observation_difference':max(m['max_abs_observation_difference'] for m in metrics),
        'maximum_reward_difference':max(m['max_abs_reward_difference'] for m in metrics),
        'all_initial_conditions_equal_within_scenario':all(m['same_initial_xk_zk'] for m in metrics),
        'all_actions_equal_within_scenario':all(m['same_action_sequence'] for m in metrics),
        'data_validation':validations,
        'interpretation':'Fixed-input deterministic reproducibility, not statistical performance superiority.'}
    write_json(output/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    print('Evidence:',output,flush=True)


if __name__=='__main__':main()
