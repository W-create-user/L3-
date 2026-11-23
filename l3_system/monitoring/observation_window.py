"""Bound rolling observation packets, without simulator calls or safety approval.

Current episode/version context must independently come from B at gate time.
This collector's last-seen version is not an authoritative live-state provider.
"""
from copy import deepcopy
import re
import numpy as np


class ObservationWindow:
    def __init__(self,binding,window_size=4):
        if type(window_size)is not int or window_size<2:raise ValueError('window size >=2 required')
        if not isinstance(binding,dict) or binding.get('schema_version')!='AnomalyBinding/1.0':raise ValueError('explicit semantic binding required')
        sizes={'Atropine':(39,4),'mAb':(1970,9)}
        if binding.get('environment') not in sizes:raise ValueError('known environment binding required')
        fingerprint=binding.get('collection_config_sha256')
        if not isinstance(fingerprint,str) or not re.fullmatch('[0-9a-f]{64}',fingerprint) or fingerprint=='0'*64:
            raise ValueError('non-placeholder collection configuration fingerprint required')
        for group,size in zip(('observation','action'),sizes[binding['environment']]):
            entries=binding.get(group)
            if not isinstance(entries,list) or len(entries)!=size:raise ValueError('binding dimensions differ')
            for i,entry in enumerate(entries):
                if (not isinstance(entry,dict) or set(entry)!={'index','name','unit','coordinate'}
                        or type(entry['index'])is not int or entry['index']!=i
                        or any(not isinstance(entry[k],str) or not entry[k].strip() for k in ('name','unit','coordinate'))):
                    raise ValueError('complete ordered variable binding required')
        self._binding=deepcopy(binding);self.window_size=window_size
        self._episode=None;self._versions=[];self._values=[];self._last_version=None;self._seen_episodes=set()

    def start_episode(self,episode_id):
        if not isinstance(episode_id,str) or not episode_id.strip() or episode_id in self._seen_episodes:
            self._clear()
            raise ValueError('new never-reused episode identity required')
        self._seen_episodes.add(episode_id)
        self._episode=episode_id;self._versions=[];self._values=[];self._last_version=None

    def _clear(self):self._versions=[];self._values=[]

    def append(self,observation,*,episode_id,state_version):
        try:
            if self._episode is None or episode_id!=self._episode:raise ValueError('explicit matching episode required')
            if type(state_version)is not int or state_version<0:raise ValueError('nonnegative integer state version required')
            if self._last_version is not None and state_version<=self._last_version:raise ValueError('duplicate or reordered state version')
            values=np.asarray(observation)
            if (values.shape!=(len(self._binding['observation']),) or values.dtype.kind not in 'fiu'
                    or not np.isfinite(values).all()):raise ValueError('finite bound numeric observation required')
        except (ValueError,TypeError):
            self._clear()
            raise ValueError('invalid/replayed observation invalidated prior window') from None
        gap=self._last_version is not None and state_version!=self._last_version+1
        if gap:self._clear()
        self._last_version=state_version;self._versions.append(state_version);self._values.append(values.tolist())
        self._versions=self._versions[-self.window_size:];self._values=self._values[-self.window_size:]
        return {'ready':len(self._versions)==self.window_size,'continuity_reset':gap,
                'collected_records':len(self._versions),'execution_authorized':False}

    def packet(self):
        if len(self._versions)!=self.window_size:raise ValueError('full fresh contiguous window unavailable')
        return {'binding':deepcopy(self._binding),'episode_id':self._episode,
                'state_versions':list(self._versions),'values':deepcopy(self._values)}
