"""Read public joint-freeze receipts; never signs another role or launches runs."""
import argparse,json
from pathlib import Path

def readiness(config):
 missing=[]
 if not config.get('formal_launch_authorized'):missing.append('A formal launch authorization')
 for role in ('A','B','C','D'):
  if not config.get('joint_signoffs',{}).get(role):missing.append(role+' signoff on same executable version')
 for key in ('integrated_commit','runner_sha256','dependency_lock_sha256','public_fixture_end_to_end_validation'):
  if not config.get('formal_runner',{}).get(key):missing.append('formal_runner.'+key)
 if not config.get('gate',{}).get('effective_primary_config_sha256'):missing.append('effective model gate configuration hash')
 if not config.get('holdout',{}).get('account_isolation_verified'):missing.append('D holdout access isolation verification')
 return {'formal_ready':not missing,'missing':missing,'blind_test_started_by_this_check':False,
 'scope':'public JSON evidence only; does not authenticate signatures or independently validate supplied hashes'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/research/a08_mab_joint_freeze_candidate_20261126.json');a=p.parse_args()
 print(json.dumps(readiness(json.loads(a.config.read_text())),indent=2))
