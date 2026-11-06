"""Reproducible engineering comparisons; no industrial performance claim."""
from pathlib import Path
import sys,json,csv,time,hashlib,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrated_system.system import EnvironmentAdapter,ExecutionAgent,FourAgentSystem,MonitoringGateway

OUT=ROOT/'evidence/four_agents'
OUT.mkdir(parents=True,exist_ok=False)
rows=[]
for index,seed in enumerate([1101,1102,1103]):
    for mode in ['single_execution','four_agents']:
        with EnvironmentAdapter() as env:
            start=time.perf_counter()
            if mode=='four_agents':
                system=FourAgentSystem(env,OUT/f'{mode}_{seed}')
                result=system.run(seed=seed)
                execution=result['execution_result']
                assert result['status']=='completed' and result['step_delta']==1
            else:
                agent=ExecutionAgent(env,log_path=OUT/f'single_{seed}.jsonl');agent.reset(seed=seed)
                monitor=MonitoringGateway(env.action_space)
                task={'task_id':f'SINGLE-{seed}','task_type':'control_cycle','description':'Matched midpoint baseline'}
                execution=agent.execute_assignment(task,monitor.check_request)
                assert execution['success']
            rows.append({'seed':seed,'mode':mode,'success':execution['success'],
                         'reward':execution['reward'],'efactor':execution['observation'][6],
                         'elapsed_seconds':time.perf_counter()-start,'action':json.dumps(execution['action'])})
for seed in [1101,1102,1103]:
    with EnvironmentAdapter() as env:
        result=FourAgentSystem(env,OUT/f'reject_{seed}').run(seed=seed,scenario='reject')
        assert result['failure_reason']=='safety_rejected' and result['step_delta']==0
with EnvironmentAdapter() as env:
    result=FourAgentSystem(env,OUT/'inspect').run(goal='检查当前仿真状态')
    assert result['status']=='completed' and result['step_delta']==0
    monitor=MonitoringGateway(env.action_space)
    rng=np.random.default_rng(1101)
    audit=[]
    for i in range(40):
        action=rng.uniform(env.action_space.low,env.action_space.high)
        should_pass=i<20
        if not should_pass:action[0]=float(env.action_space.high[0])+1
        request={'task_id':f'AUDIT-{i}','request_id':f'AUDIT-REQ-{i}','action':action.tolist()}
        review=monitor.check_request(request,None)
        audit.append({'id':i,'label':'within_model_bounds' if should_pass else 'outside_model_bounds',
                      'action':json.dumps(request['action']),'monitored_decision':review['decision'],
                      'no_monitor_policy':'would_forward_without_review_NOT_executed',
                      'rule_id':review['rule_id']})
    assert all(r['monitored_decision']==('pass' if i<20 else 'reject') for i,r in enumerate(audit))
for name,data in [('single_vs_four.csv',rows),('monitor_offline.csv',audit)]:
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
paired=[{'seed':seed,'reward_difference':rows[i*2+1]['reward']-rows[i*2]['reward'],
         'efactor_difference':rows[i*2+1]['efactor']-rows[i*2]['efactor']} for i,seed in enumerate([1101,1102,1103])]
summary={'status':'PASS','normal_four_agent_runs':3,'rejected_four_agent_runs':3,'inspect_runs':1,
         'real_steps_in_this_experiment':6,'single_baseline_runs':3,'paired':paired,
         'offline_monitor_cases':40,'within_bounds_passed':20,'outside_bounds_rejected':20,
         'offline_no_monitor_executed':False,'bid_cost_origin':'configured unit cost 1.0; capability exact-match indicator; single eligible worker per role',
         'limits':'n=3 engineering verification; not proof of 90% industrial completion, 20% efficiency improvement, multi-executor CNP superiority or physical process safety',
         'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'main.py',ROOT/'integrated_system/system.py',ROOT/'决策智能体/agents/decision_agent.py',ROOT/'C_member_before_20261108/agents/scheduling_agent.py',ROOT/'agents/monitoring_agent.py',ROOT/'执行智能体/code/integration_abd_1103/execution_agent.py']}}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
for mode in ['single_execution','four_agents']:
    data=[r for r in rows if r['mode']==mode]
    axes[0].plot([r['seed'] for r in data],[r['efactor'] for r in data],marker='o',label=mode)
    axes[1].plot([r['seed'] for r in data],[r['elapsed_seconds'] for r in data],marker='o',label=mode)
axes[0].set_ylabel('E-factor after one step');axes[1].set_ylabel('Measured wall time (seconds)')
for ax in axes:ax.set_xlabel('Reset seed');ax.set_xticks([1101,1102,1103]);ax.legend();ax.grid(alpha=.2)
fig.suptitle('Matched midpoint control: n=3, engineering verification only')
fig.tight_layout();fig.savefig(OUT/'single_vs_four.png',dpi=150);plt.close(fig)
print(json.dumps(summary,ensure_ascii=True))
