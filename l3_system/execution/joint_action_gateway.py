"""One in-process execution owner for a coupled model; no worker may own an adapter.

The ledger is process-local, not a durable distributed exactly-once protocol.
D rejection or uncertain execution latches takeover until an explicit reset acknowledgement.
"""
from __future__ import annotations
import copy
import math
import threading
import time
import numpy as np

class JointActionGateway:
    def __init__(self, executor, monitor, config, *, clock=time.monotonic):
        self.executor=executor;self.adapter=executor.adapter;self.monitor=monitor
        if getattr(self.adapter,'_joint_gateway_owner',None) is not None:raise RuntimeError('Adapter already has a joint execution owner')
        if getattr(monitor,'demo_mode',True) is not False:raise ValueError('Formal D monitor required')
        self.config=copy.deepcopy(config);self.clock=clock;self._lock=threading.RLock()
        self._active=None;self._rounds={};self._results={};self.takeover_required=False
        self._execution_thread_id=threading.get_ident()
        self._latched_episode=None
        meta=getattr(self.adapter,'metadata',{})
        if config.get('environment_id')!=meta.get('environment_id'):raise ValueError('Backend mismatch')
        if config.get('action_coordinate')!=meta.get('action_coordinate'):raise ValueError('Action coordinate mismatch')
        if tuple(self.adapter.action_space.shape)!=(config['action_size'],):raise ValueError('Action dimension mismatch')
        if tuple(self.adapter.observation_space.shape)!=(config['observation_size'],):raise ValueError('Observation dimension mismatch')
        self.partitions={}
        covered=[]
        for row in config['partitions']:
            actor=row['actor_id'];indices=row['action_indices'];observed=row['observation_indices']
            if (not isinstance(actor,str) or not actor or actor in self.partitions or not indices
                    or any(type(i)is not int or not 0<=i<config['action_size'] for i in indices)
                    or any(type(i)is not int or not 0<=i<config['observation_size'] for i in observed)):
                raise ValueError('Invalid partition')
            self.partitions[actor]=copy.deepcopy(row);covered+=indices
        if sorted(covered)!=list(range(config['action_size'])):raise ValueError('Partitions overlap or do not cover action axes')
        self.adapter._joint_gateway_owner=self
    def _now(self):
        value=self.clock()
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise RuntimeError('Invalid monotonic clock')
        return value
    def _context(self):
        return self.adapter.state_version,self.adapter.metadata.get('episode_id')
    def _finish_record(self, cycle_id, status, *, result=None, joint=None, step_delta=0, reason=None):
        row=self._rounds[cycle_id]
        success=bool(result and result.get('success') is True and result.get('executed') is True and step_delta==1)
        uncertain=step_delta!=0 and not success
        if uncertain or (result and result.get('execution_status')=='rejected'):
            self.takeover_required=True;self._latched_episode=row['episode_id']
        record={'cycle_id':cycle_id,'status':status,'finalized':True,'success':success,
                'requested_joint_action':joint.tolist() if joint is not None else None,
                'executed_joint_action':copy.deepcopy(result.get('action')) if result and result.get('executed') is True else None,
                'executed':bool(result.get('executed')) if result else (False if step_delta==0 else None),
                'step_called':step_delta!=0,'step_delta':step_delta,'reward':result.get('reward') if result else None,
                'rejected':bool(result and result.get('execution_status')=='rejected'),
                'reason':reason or (result.get('failure_reason') if result else status),
                'observation_before':copy.deepcopy(row['observation']),
                'observation_after':copy.deepcopy(result.get('observation')) if result else (copy.deepcopy(row['observation']) if step_delta==0 and self._context()==(row['state_version'],row['episode_id']) else None),
                'step_calls_since_begin':self.adapter.step_calls-row['step_calls'],
                'terminated':result.get('terminated',False) if result else None,
                'truncated':result.get('truncated',False) if result else None,
                'state_version_before':row['state_version'],'state_version_after':self.adapter.state_version,
                'episode_id':row['episode_id'],'transition_complete':success,
                'learning_eligible':False,'C05_contract_confirmed':False,
                'takeover_required':self.takeover_required,'execution_result':copy.deepcopy(result)}
        self._results[cycle_id]=record;self._active=None
        return copy.deepcopy(record)
    def begin(self, cycle_id, *, timeout_seconds=5):
        with self._lock:
            if not isinstance(cycle_id,str) or not cycle_id:raise ValueError('Cycle ID required')
            if cycle_id in self._rounds:raise ValueError('Cycle ID has already been used')
            if isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float)) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:
                raise ValueError('Finite positive timeout required')
            if self.takeover_required:raise RuntimeError('Unknown/failed execution requires reset and human acknowledgement')
            self.adapter.require_ready()
            if self._active is not None:raise RuntimeError('Finish or abort the active cycle first')
            if self.executor._pending:raise RuntimeError('Outstanding non-gateway request')
            version,episode=self._context()
            if not episode:raise ValueError('Trusted adapter episode identity required')
            obs=self.adapter.get_observation()
            self._rounds[cycle_id]={'state_version':version,'episode_id':episode,'deadline':self._now()+timeout_seconds,
                                    'observation':copy.deepcopy(obs),'step_calls':self.adapter.step_calls,'components':{}}
            self._active=cycle_id
            return {'cycle_id':cycle_id,'state_version':version,'episode_id':episode,
                    'local_observations':{actor:[obs[i] for i in spec['observation_indices']] for actor,spec in self.partitions.items()},
                    'central_state':copy.deepcopy(obs),'coordinate':self.config['action_coordinate'],'training_enabled':False}
    def _valid_open(self, cycle_id):
        if cycle_id not in self._rounds:raise ValueError('Unknown cycle')
        if cycle_id in self._results:raise RuntimeError('Cycle already closed')
        row=self._rounds[cycle_id]
        if self._now()>=row['deadline']:
            return self._finish_record(cycle_id,'timeout',reason='Missing/late components; no default actions are filled')
        if self._context()!=(row['state_version'],row['episode_id']):
            return self._finish_record(cycle_id,'stale_round',reason='Environment changed since begin')
        return None
    def submit(self, cycle_id, actor_id, action, *, state_version, episode_id, coordinate):
        with self._lock:
            closed=self._valid_open(cycle_id)
            if closed:return closed
            row=self._rounds[cycle_id]
            try:
                if type(state_version)is not int or (state_version,episode_id)!=(row['state_version'],row['episode_id']):raise ValueError('Stale component context')
                if coordinate!=self.config['action_coordinate']:raise ValueError('Component action coordinate mismatch')
                if actor_id not in self.partitions:raise ValueError('Unknown actor')
                if actor_id in row['components']:raise ValueError('Duplicate component taints this cycle')
                value=np.asarray(action)
                if value.dtype.kind not in 'fiu' or value.shape!=(len(self.partitions[actor_id]['action_indices']),) or not np.isfinite(value).all():
                    raise ValueError('Invalid component shape/type/finite values')
                # Preserve out-of-range finite candidates for D; no hidden clipping.
                row['components'][actor_id]=value.astype(float).copy()
            except Exception:
                self._finish_record(cycle_id,'invalid_component')
                raise
            return {'status':'accepted','finalized':False,'missing_actors':sorted(set(self.partitions)-row['components'].keys())}
    def finish(self, cycle_id):
        with self._lock:
            if cycle_id in self._results:return copy.deepcopy(self._results[cycle_id])
            closed=self._valid_open(cycle_id)
            if closed:return closed
            row=self._rounds[cycle_id]
            missing=set(self.partitions)-row['components'].keys()
            if missing:return {'status':'waiting','finalized':False,'missing_actors':sorted(missing),'learning_eligible':False}
            if threading.get_ident()!=self._execution_thread_id:
                raise RuntimeError('Physical execution must run on the gateway owner thread; workers only submit components')
            joint=np.empty(self.adapter.action_space.shape,dtype=float)
            for actor,part in self.partitions.items():joint[part['action_indices']]=row['components'][actor]
            before=self.adapter.step_calls
            task={'task_id':'joint:'+cycle_id,'task_type':'control_cycle','description':'B-owned coupled model cycle'}
            def checker(request,observation):
                if (request['state_version'],request.get('episode_id'))!=(row['state_version'],row['episode_id']):
                    raise RuntimeError('Context changed before D review')
                if self._context()!=(row['state_version'],row['episode_id']):raise RuntimeError('External step/reset detected')
                review=self.monitor.check_request(request,observation)
                if self._now()>=row['deadline']:raise TimeoutError('Dispatch deadline exceeded during D review')
                if self._context()!=(row['state_version'],row['episode_id']):raise RuntimeError('State changed during D review')
                return review
            try:
                outcome=self.executor.execute_assignment(task,checker,action=joint)
                delta=self.adapter.step_calls-before
                if outcome.get('success') is True and delta!=1:raise RuntimeError('Unexpected physical execution count')
                status='completed' if outcome.get('success') is True else outcome['execution_status']
                return self._finish_record(cycle_id,status,result=outcome,joint=joint,step_delta=delta)
            except Exception as error:
                for req_id,pending in list(self.executor._pending.items()):
                    if pending.get('task_id')==task['task_id']:self.executor.cancel_action_request(req_id,'gateway failure')
                return self._finish_record(cycle_id,'execution_error',joint=joint,step_delta=self.adapter.step_calls-before,
                                           reason=type(error).__name__+': '+str(error))
    def abort(self,cycle_id,reason='coordinator_cancelled'):
        with self._lock:
            if cycle_id in self._results:return copy.deepcopy(self._results[cycle_id])
            if cycle_id not in self._rounds:raise ValueError('Unknown cycle')
            return self._finish_record(cycle_id,'cancelled',reason=reason)
    def acknowledge_after_reset(self,*,operator,reason):
        with self._lock:
            if not self.takeover_required:raise RuntimeError('No pending takeover')
            if not all(isinstance(v,str) and v.strip() for v in [operator,reason]):raise ValueError('Human identity/reason required')
            self.adapter.require_ready()
            if self.adapter.metadata.get('episode_id')==self._latched_episode or self.executor._pending:
                raise RuntimeError('A new explicit episode and no outstanding requests are required')
            self.takeover_required=False;self._latched_episode=None
