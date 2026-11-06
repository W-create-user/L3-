"""Map D v1.1's approved flag to B's request-bound SafetyDecision."""
import copy

from .environment_adapter import to_plain


def normalize_safety_decision(request, review):
    if not isinstance(review, dict):
        raise ValueError('A MonitoringAgent review dictionary is required')
    raw = to_plain(copy.deepcopy(review))
    if raw.get('request_id') is not None and raw['request_id'] != request['request_id']:
        raise ValueError('Monitoring review request_id does not match ActionRequest')
    if raw.get('task_id') is not None and raw['task_id'] != request['task_id']:
        raise ValueError('Monitoring review task_id does not match Task')
    if 'approved' in raw:
        if type(raw['approved']) is not bool:
            raise ValueError('Monitoring approved must be a real boolean, not a string or number')
        approved = raw['approved']
    elif raw.get('decision') in ('pass', 'reject'):
        # D check_request returns task_id+decision without request_id.
        # Accept it in the synchronous context of this exact pending request.
        if (raw.get('request_id') != request['request_id']
                and raw.get('task_id') != request['task_id']):
            raise ValueError('Decision review must match request_id or task_id')
        approved = raw['decision'] == 'pass'
    else:
        raise ValueError('Monitoring review requires approved: bool')
    if 'decision' in raw and raw['decision'] != ('pass' if approved else 'reject'):
        raise ValueError('Monitoring decision conflicts with approved')
    if 'status' in raw:
        if not isinstance(raw['status'], str) or raw['status'].lower() != ('pass' if approved else 'reject'):
            raise ValueError('Monitoring status conflicts with approved')
    reason = raw.get('reason')
    if not approved and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Monitoring rejection must include a nonempty reason')
    return {
        'schema_version': 'SafetyDecision/1.0',
        'request_id': request['request_id'],
        'task_id': request['task_id'],
        'subtask_id': request.get('subtask_id'),
        'approved': approved,
        'decision': 'pass' if approved else 'reject',
        'reason': reason or 'Monitoring approved this candidate action',
        'level': raw.get('level', raw.get('risk_level')),
        'risk_level': raw.get('risk_level', raw.get('level')),
        'rule_id': raw.get('rule_id'),
        'intervention': raw.get('intervention'),
        'demo_mode': raw.get('demo_mode'),
        'raw_review': raw.get('raw_review', raw),
    }
