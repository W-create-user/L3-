"""Sensor transformations and causal modalities; never mutate a simulator state."""
import math
import numpy as np

ALLOWED_FAULTS={'bias','noise','missing'}

def sensor_view(observation, *, index, step, fault, rng):
    original=np.asarray(observation)
    if original.ndim!=1 or original.dtype.kind not in 'fiu' or not np.isfinite(original).all():raise ValueError('Finite simulator observation required')
    if type(index)is not int or not 0<=index<len(original):raise ValueError('Feature index invalid')
    delivered=original.astype(float).tolist();mask=[False]*len(delivered)
    active=False;event=None
    if fault is not None:
        if fault['kind'] not in ALLOWED_FAULTS:raise ValueError('Unknown sensor corruption')
        first,last=fault['first_step'],fault['last_step']
        if type(first)is not int or type(last)is not int or not 1<=first<=last:raise ValueError('Invalid schedule')
        active=first<=step<=last
        if active:
            before=delivered[index];kind=fault['kind']
            if kind=='bias':
                magnitude=fault['multiplier']
                if isinstance(magnitude,bool) or not math.isfinite(magnitude) or magnitude==1:raise ValueError('Invalid bias magnitude')
                delivered[index]=float(before*magnitude)
            elif kind=='noise':
                sigma=fault['sigma']
                if isinstance(sigma,bool) or not math.isfinite(sigma) or sigma<=0:raise ValueError('Invalid noise sigma')
                delivered[index]=float(before+rng.normal(0,sigma))
            else:delivered[index]=None;mask[index]=True
            event={'kind':'synthetic_sensor_'+kind,'observation_index':index,'before':float(before),
                   'after':delivered[index],'application':'post_step_detector_view_only','source':'declared_seeded_injection_schedule'}
    return delivered,mask,{'sensor_fault_label':int(active),'sensor_fault_kind':fault['kind'] if active else None,
                          'physical_health_label':None,'label_source':'known_injection_not_threshold_or_reward'},event

def feature_payload(delivered,mask,*,indices,scale,unit):
    values=[delivered[i] for i in indices];missing=[mask[i] for i in indices]
    value=values[-1]
    # No split/seed/episode IDs, labels, injection descriptions or future samples in model text.
    text='measurement unavailable' if value is None else f'measurement {value*scale:.6g} {unit}'
    return {'series':values,'missing_mask':missing,'text':text,
            'text_provenance':'derived_from_current_delivered_values_not_independent_information'}
