"""Offline evaluation of ONE unapproved Atropine experiment candidate; never authorizes execution."""
import math


def evaluate(observation, *, backend, normalize, layout):
    result={'runtime_enabled':False,'threshold_approval':'pending_A_team',
            'candidate':'Atropine_Efactor_relative_high','operational_decision':None}
    if backend!='AtropineEnvGym' or normalize is not False or layout!='atropine-default39-v1':
        return dict(result,status='incompatible_configuration')
    try:
        if len(observation)!=39 or any(not math.isfinite(float(x)) for x in observation):
            return dict(result,status='invalid_input')
        reference,current=float(observation[5]),float(observation[6])
    except (TypeError,ValueError,OverflowError):
        return dict(result,status='invalid_input')
    # Box ceiling is used only to detect a hidden/censored threshold, not as a safety limit.
    if reference<=0 or 1.10*reference>=20:
        return dict(result,status='unsupported_reference_or_censoring')
    return dict(result,status='candidate_breach' if current>1.10*reference else 'within_candidate_band',
                observed=current,reference=reference,proposed_upper=1.10*reference,
                relative_change=current/reference-1)
