"""Inventory existing B logs without reclassifying archive records as training data."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(path):
    path=Path(path)
    raw=path.read_bytes()
    def invalid(value):raise ValueError(f'nonfinite JSON constant: {value}')
    rows=[json.loads(line,parse_constant=invalid) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    errors=[];pending={};results=set();resets=[];calls=0;rejected=0;outcomes=0
    for number,row in enumerate(rows,1):
        if not isinstance(row,dict):
            errors.append(f'line {number}: event must be an object');continue
        event=row.get('event')
        if event=='environment_reset':
            if pending:errors.append(f'line {number}: reset abandoned unresolved prepared requests')
            resets.append({'seed':row.get('seed'),'observation_dimension':len(row['observation'])
                           if isinstance(row.get('observation'),list) else None,
                           'state_version':row.get('state_version')})
            pending={}
        elif event=='action_prepared':
            if not resets:errors.append(f'line {number}: prepared action precedes any reset record')
            request=row.get('action_request',{})
            request_id=request.get('request_id') if isinstance(request,dict) else None
            if not isinstance(request_id,str) or not request_id:
                errors.append(f'line {number}: missing prepared request_id')
            elif request_id in pending or request_id in results:
                errors.append(f'line {number}: duplicate prepared request_id')
            else:
                pending[request_id]=request
        elif event=='execution_result':
            outcome=row.get('result',{})
            if not isinstance(outcome,dict):
                errors.append(f'line {number}: malformed result');continue
            rid=outcome.get('request_id');outcomes+=1
            if not isinstance(rid,str) or not rid:
                errors.append(f'line {number}: missing/invalid result request_id');continue
            if rid in results:errors.append(f'line {number}: duplicate result request_id')
            results.add(rid)
            request=pending.pop(rid,None)
            if request is None:errors.append(f'line {number}: result has no prepared request in current episode')
            elif request.get('task_id')!=outcome.get('task_id'):
                errors.append(f'line {number}: result task_id differs from prepared request')
            if type(outcome.get('executed')) is not bool:
                errors.append(f'line {number}: executed must be boolean')
            calls+=outcome.get('executed') is True
            rejected+=outcome.get('execution_status')=='rejected'
            if outcome.get('executed') is True and outcome.get('execution_status')=='rejected':
                errors.append(f'line {number}: rejected outcome marked executed')
    if pending:errors.append('prepared requests without recorded result remain')
    return {'path':path.as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'event_records':len(rows),
            'reset_records':len(resets),'reset_details':resets,'outcome_records':outcomes,
            'executed_calls_reported':calls,'rejected_candidates':rejected,'consistency_errors':errors,
            'training_ready':False,
            'missing_for_D01_intake':['D01Step normalized records with verified simulation-time units',
                                     'reviewed physical variable dictionary','independent anomaly labels and split review',
                                     'dataset provenance and permission review']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path,nargs='+')
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.code_commit):parser.error('code-commit must be a full Git SHA')
    reports=[]
    for path in args.logs:
        try:reports.append(audit(path))
        except (OSError,ValueError,UnicodeError) as exc:
            reports.append({'path':path.as_posix(),'training_ready':False,'consistency_errors':[type(exc).__name__]})
    result={'kind':'archive_structure_audit_not_new_simulation','code_commit':args.code_commit,'files':reports,
            'independent_training_episode_count':None,
            'limits':'Reset and execution counts do not establish independent samples or process safety.'}
    with args.report.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'files':len(reports),'files_with_consistency_errors':sum(bool(r['consistency_errors']) for r in reports),
                      'training_ready':False},ensure_ascii=False))
    raise SystemExit(1 if any(r['consistency_errors'] for r in reports) else 0)


if __name__=='__main__':main()
