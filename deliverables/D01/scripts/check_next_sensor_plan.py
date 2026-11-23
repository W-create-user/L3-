"""Validate a non-executable next-sensor benchmark plan; no acquisition."""
import argparse,json
from pathlib import Path


def validate(plan):
    if not isinstance(plan,dict) or plan.get('schema_version')!='SensorBenchmarkNext/0.1':raise ValueError('known plan required')
    if plan.get('status')!='draft' or plan.get('execution_authorized') is not False:raise ValueError('plan must remain an unexecuted draft')
    rows=plan.get('episodes')
    if not isinstance(rows,list) or len(rows)!=32 or plan.get('horizon_steps')!=60:raise ValueError('32 planned episodes of 60 steps required')
    expected_seeds={'train':set(range(16001,16009)),'validation':set(range(17001,17013)),'test':set(range(18001,18013))}
    ids=set();seeds=set();groups={};counts={s:{'clean':0,'fault':0} for s in ('train','validation','test')};schedules=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('episode object required')
        split=row.get('split');eid=row.get('episode_id');group=row.get('group_id');seed=row.get('seed')
        if (split not in counts or not isinstance(eid,str) or not eid or eid in ids
                or not isinstance(group,str) or not group or type(seed)is not int or seed not in expected_seeds.get(split,set()) or seed in seeds
                or row.get('reset_count')!=1 or row.get('horizon_steps')!=60 or row.get('physical_health_label') is not None):
            raise ValueError('distinct episode/seed, scope and fixed full horizon required')
        if group in groups and groups[group]!=split:raise ValueError('lineage cannot cross splits')
        groups[group]=split;ids.add(eid);seeds.add(seed);fault=row.get('fault')
        if fault is None:counts[split]['clean']+=1
        else:
            if (split=='train' or not isinstance(fault,dict) or fault.get('observation_index')!=6
                    or fault.get('kind')!='multiplicative_bias' or fault.get('application')!='post_step_delivered_only'):
                raise ValueError('only declared post-step sensor corruption; never corrupt training or process state')
            first,last,mult=(fault.get(k) for k in ('first_step','last_step','multiplier'))
            if type(first)is not int or type(last)is not int or not 8<=first<=last<=57 or type(mult) not in (int,float) or not .5<=mult<=1.5 or mult==1:
                raise ValueError('bounded predeclared injection schedule required')
            counts[split]['fault']+=1;schedules.append((first,last,mult))
    if counts!={'train':{'clean':8,'fault':0},'validation':{'clean':6,'fault':6},'test':{'clean':6,'fault':6}}:
        raise ValueError('explicit whole-clean validation/test controls required')
    if len({s[0] for s in schedules})<3 or len({s[1]-s[0]+1 for s in schedules})<3 or len({s[2] for s in schedules})<3:
        raise ValueError('vary onset, duration and amplitude before collection')
    missing=[k for k in ('collector_review_ref','budget_confirmation_ref') if not plan.get(k)]
    return {'structure_valid':True,'planned_steps':1920,'counts':counts,'missing_confirmations':missing,
            'execution_authorized':False,'test_unopened_claim_verified':False,'data_collected':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);a=p.parse_args()
    print(json.dumps(validate(json.loads(a.plan.read_text())),indent=2))
