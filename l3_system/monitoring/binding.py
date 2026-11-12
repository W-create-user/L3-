"""Exact model/data binding; dimensions alone do not establish identity."""
from copy import deepcopy
import hashlib
import json


def fingerprint(binding):
    return hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_binding(dictionary,config_sha256):
    if dictionary['environment'] not in ('Atropine','mAb'):raise ValueError('unknown environment')
    fields={}
    for group in ('observation','action'):
        fields[group]=[{k:deepcopy(v[k]) for k in ('index','name','unit','coordinate')} for v in dictionary[group]]
        if [v['index'] for v in fields[group]]!=list(range(len(fields[group]))):raise ValueError('ordered variable indices required')
    return dict(schema_version='AnomalyBinding/1.0',environment=dictionary['environment'],
                collection_config_sha256=config_sha256,**fields)


def check_window(packet,context,expected,window_size):
    if not isinstance(packet,dict) or not isinstance(context,dict):raise ValueError('trusted bound window/context required')
    if packet.get('binding')!=expected or context.get('binding')!=expected:raise ValueError('model/live semantic binding mismatch')
    episode=context.get('episode_id');version=context.get('state_version')
    if not isinstance(episode,str) or not episode.strip() or type(version) is not int or version<0:
        raise ValueError('trusted episode and state version required')
    if packet.get('episode_id')!=episode:raise ValueError('window belongs to another episode')
    versions=packet.get('state_versions')
    if (not isinstance(versions,list) or len(versions)!=window_size
            or any(type(v) is not int or v<0 for v in versions)
            or versions!=list(range(version-window_size+1,version+1))):
        raise ValueError('window must contain consecutive current-episode state versions')
    return packet['values']
