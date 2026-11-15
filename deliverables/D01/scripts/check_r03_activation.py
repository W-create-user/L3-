"""Offline readiness checklist only; cannot enable/export a runtime rule."""
import argparse,json,math
from pathlib import Path


def check(review):
    if not isinstance(review,dict) or review.get('schema_version')!='R03ActivationReview/0.1':raise ValueError('known activation review required')
    if review.get('status')!='draft' or review.get('runtime_enabled') is not False:raise ValueError('this tool does not accept enabled or approved configurations')
    if review.get('environment')!='Atropine' or review.get('rule_kind')!='relative_e_factor_candidate':raise ValueError('wrong environment/rule scope')
    fields=('approved_operating_config_ref','goal_contract_ref','threshold_basis_ref','team_confirmation_ref',
            'disposition_ref','recovery_ref','fresh_context_contract_ref','boundary_test_record_ref','rollback_version_ref')
    missing=[k for k in fields if not isinstance(review.get(k),str) or not review[k].strip()]
    ratio=review.get('relative_upper')
    if ratio is None:missing.append('relative_upper')
    elif type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio<=0:raise ValueError('explicit finite positive experimental ratio required')
    return {'missing_fields':missing,'declarations_complete':not missing,'approval_verified':False,
            'runtime_enabled':False,'export_allowed':False,
            'next_step':'independent review of references, boundary tests and explicit versioned runtime change; this check is not approval'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('review',type=Path);a=p.parse_args()
    print(json.dumps(check(json.loads(a.review.read_text())),indent=2))
