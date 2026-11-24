"""D07 deterministic paired run ledger and content-verified readiness review.

Preparation does not launch trials or invent missing goal/control definitions.
"""
import hashlib
import json
from pathlib import Path
import random


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(plan, root):
    from deliverables.D01.scripts.validate_experiment_draft import validate
    root = Path(root).resolve()
    review = validate(plan)
    if review['errors']:
        raise ValueError('; '.join(review['errors']))
    seed = plan.get('run_order_seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('explicit run order seed required')
    rows = []
    for condition in plan['conditions']:
        for pair_seed in condition['pair_seeds']:
            for arm in plan['arms']:
                rows.append({'run_id':f"{condition['id']}-{pair_seed}-{arm}",
                    'pair_id':f"{condition['id']}-{pair_seed}", 'condition':condition['id'],
                    'reset_seed':pair_seed, 'arm':arm, 'environment':plan['environment'],
                    'status':'planned', 'started':False, 'goal_met':None,
                    'deadline_met':None, 'safety_violations':None, 'simulated_makespan':None,
                    'failure_reason':None, 'exclusion_reason':None, 'execution_authorized':False})
    random.Random(seed).shuffle(rows)
    for i,row in enumerate(rows,1): row['run_order'] = i
    references = {}
    blockers = list(review['missing_confirmations'])
    # Only repository-relative files are accepted; strings do not count as signatures.
    for field in ('goal_contract_ref','controller_definitions_ref','resource_registry_ref','approval_record_ref'):
        value = plan.get(field)
        if not value: continue
        path = (root/value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            blockers.append(field+':file_missing_or_outside_repository')
        else: references[field] = {'path':value, 'sha256':digest(path)}
    for condition in plan['conditions']:
        value = condition.get('definition_ref')
        if not value: continue
        path = (root/value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            blockers.append('condition:'+condition['id']+':unverified_reference')
        else: references['condition:'+condition['id']] = {'path':value, 'sha256':digest(path)}
    goal_ref = references.get('goal_contract_ref')
    if goal_ref:
        goal = json.loads((root/goal_ref['path']).read_text())
        if goal.get('environment_id') != plan['environment']:
            blockers.append('goal_environment_mismatch')
        # This published A target is an engineering regression target.
        if goal_ref['path'] == 'configs/goals/atropine_goal_regression.json':
            blockers.append('engineering_regression_goal_not_production_acceptance')
    return {'schema_version':'D07Readiness/1.0','run_count':len(rows),
            'paired_condition_seed_groups':len(rows)//len(plan['arms']),
            'run_order_seed':seed, 'reference_sha256':references,
            'blockers':sorted(set(blockers)), 'formal_experiment_ready':False,
            'execution_authorized':False, 'frozen':False, 'ledger':rows,
            'limits':'D preparation only. External approvals are not authenticated by this checker; planned runs are not measurements.'}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=prepare(json.loads(args.plan.read_text()),Path(__file__).resolve().parents[2])
    args.output.mkdir(parents=True,exist_ok=False)
    ledger=result.pop('ledger')
    (args.output/'run_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    result['plan_sha256']=digest(args.plan)
    result['ledger_sha256']=digest(args.output/'run_ledger.json')
    (args.output/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'planned_runs':len(ledger),'formal_experiment_ready':False,'blockers':result['blockers']}))
