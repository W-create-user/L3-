"""Recompute the private B04/B05/B06 acceptance summary from actual artifacts."""
import argparse
import json
from pathlib import Path
import re
from data_generation.validation import validate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['b04','mpc','dataset','reproduction','gateway','tests','output']:p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    def read(folder):return json.loads((folder/'summary.json').read_text())
    study=read(a.b04);mpc=read(a.mpc);gateway=read(a.gateway)
    if not study['success'] or not mpc['success'] or not gateway['success']:raise ValueError('A live acceptance failed')
    cases={r['name']:r for r in study['cases']}
    for name in ['PI_train','PI_holdout_positive','PI_holdout_negative']:
        if not cases[name]['goal_achieved'] or not cases[name]['all_C_locks_released']:raise ValueError('PI goal/lock evidence incomplete')
    for name in ['source_steady','space_midpoint']:
        if name not in cases:raise ValueError('Baseline failures must not be removed')
    if len({cases[n]['initial_state_sha256'] for n in ['PI_train','source_steady','space_midpoint']})!=1:raise ValueError('Baseline initial states not paired')
    if mpc['initial_state_sha256']!=cases['PI_train']['initial_state_sha256']:raise ValueError('MPC initial state mismatch')
    if not mpc['solver_statuses'] or len(mpc['solver_statuses'])!=mpc['step_calls'] or not all(s=='Solve_Succeeded' for s in mpc['solver_statuses']):raise ValueError('Failed MPC solver action was released')
    original=validate_dataset(a.dataset);repeated=validate_dataset(a.reproduction)
    first=json.loads((a.dataset/'manifest.json').read_text());second=json.loads((a.reproduction/'manifest.json').read_text())
    if first['freeze_sha256']!=second['freeze_sha256'] or original['semantic_sha256']!=repeated['semantic_sha256']:raise ValueError('Frozen dataset did not reproduce')
    log=a.tests.read_text();match=re.search(r'^Ran (\d+) tests',log,re.M)
    if not match or not re.search(r'^OK$',log,re.M) or re.search(r'^FAILED',log,re.M):raise ValueError('Unit test result missing or failed')
    if gateway['training_ready'] or gateway['C05_contract_confirmed']:raise ValueError('Missing C05 cannot be called accepted')
    summary={'validated':True,'unit_tests_passed':int(match.group(1)),
             'B04_research_PI_goal_cases':3,'B04_MPC_scope':mpc['policy'],
             'B04_baseline_negative_results_preserved':True,'B05':original,'B05_reproduced':True,
             'B05_freeze_sha256':first['freeze_sha256'],'B06_gateway_checks':gateway['checks'],
             'B06_training_ready':False,'B06_C05_signoff_pending':True,
             'physical_health_labels_confirmed':False,'R03_runtime_changed':False,
             'publication_permissions_checked':False,'storage_scope':'only caller-supplied output path; no claim of D-drive archival'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
