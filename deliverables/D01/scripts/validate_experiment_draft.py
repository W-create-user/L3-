"""D07 draft consistency only. Never freezes or authorizes experiments."""
import argparse
import json
from pathlib import Path


def validate(plan):
    errors=[];missing=[]
    def check(ok,message):
        if not ok:errors.append(message)
    if not isinstance(plan,dict):return {'errors':['plan must be object'],'missing_confirmations':[],'ready_to_freeze':False}
    check(plan.get('schema_version')=='D07ExperimentDraft/0.1','unsupported schema')
    check(plan.get('status')=='draft' and plan.get('execution_authorized') is False,'draft cannot authorize or freeze experiments')
    check(plan.get('environment')=='AtropineEnvGym','environment scope changed')
    check(plan.get('arms')==['traditional_controller','single_agent','four_agent'],'three planned arms required')
    check(type(plan.get('episodes_per_condition_per_arm'))is int and plan['episodes_per_condition_per_arm']==10,'planned sample budget changed')
    conditions=plan.get('conditions');seen=set()
    if not isinstance(conditions,list) or len(conditions)!=3:errors.append('three declared conditions required')
    else:
        check([x.get('id') for x in conditions if isinstance(x,dict)]==['nominal','sensor_fault','process_disturbance'],'condition identities changed')
        for condition in conditions:
            if not isinstance(condition,dict):errors.append('condition must be object');continue
            seeds=condition.get('pair_seeds')
            if (not isinstance(seeds,list) or len(seeds)!=10 or any(type(x)is not int or x<0 for x in seeds)
                    or len(set(seeds))!=10):errors.append('ten distinct integer pair seeds required');continue
            check(not seen.intersection(seeds),'condition seed lists overlap');seen.update(seeds)
            if not condition.get('definition_ref'):missing.append('condition:'+str(condition.get('id')))
    learning=plan.get('independent_learning_seeds')
    check(isinstance(learning,list) and len(learning)==3 and all(type(s)is int and s>=0 for s in learning)
          and len(set(learning))==3,'three independent declared learning seeds required')
    for field in ('goal_contract_ref','controller_definitions_ref','resource_registry_ref','approval_record_ref'):
        if not isinstance(plan.get(field),str) or not plan[field].strip():missing.append(field)
    for field in ('simulation_horizon_records','wall_clock_budget_seconds','run_order_seed'):
        value=plan.get(field)
        if value is None:missing.append(field)
        else:check(type(value)is int and value>=(0 if field=='run_order_seed' else 1),'invalid '+field)
    confirmations=plan.get('required_confirmations')
    required={'A_goal_and_recovery','B_conditions_controls_and_budget','C_resources_and_scheduling','D_labels_splits_and_metrics'}
    if not isinstance(confirmations,dict) or set(confirmations)!=required:errors.append('A/B/C/D confirmation fields required')
    else:
        for key,value in confirmations.items():
            if not isinstance(value,str) or not value.strip():missing.append(key)
    check(plan.get('power_guarantee') is False,'sample count is not a power guarantee')
    for field in ('primary_metric','secondary_metrics','exclusion_policy','detector_policy','safety_policy','analysis_policy','pairing'):
        check(bool(plan.get(field)),'missing metric/policy: '+field)
    return {'errors':errors,'missing_confirmations':missing,'ready_to_freeze':False,
            'review_complete_claimed':False,'scope':'draft structure check; references are not independent approval verification'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path);args=parser.parse_args()
    result=validate(json.loads(args.plan.read_text()));print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(bool(result['errors']))
