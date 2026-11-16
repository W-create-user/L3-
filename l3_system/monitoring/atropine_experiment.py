"""D-owned opt-in Atropine experiment facade; leaves A/B/C source untouched.

Authorization: PR9 / 611a81fd4c78d206bf5f274261e0e713d8f10eb9.
1.10 is an isolated simulation parameter, never a physical safety limit.
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.execution.local_policy import space_midpoint

AUTHORIZATION='611a81fd4c78d206bf5f274261e0e713d8f10eb9'
APPROVED_A_RUNTIME_SHA256='1c6ecd31f61857196126eb8c9eb0d015856f2007704b95681fd1f20e210bf5e2'
SOURCE_PINS={'smpl/envs/atropineenv.py': '83ddd39a3d5a738237a915695ca4f594052a90e47e0578295af245ba4fa3a311', 'smpl/envs/models/atropine_process.py': '5165b6bf0556978af27cac7d810d5e37108fa64baccdd4fbdb2d416c9b91e88a', 'smpl/envs/_runtime_utils.py': '3afc4129c147edd04ebf7900b3a1374166857631ae09e2ecd080b4ba0b49344e', 'smpl/envs/utils.py': '0ea4f66ec000466a9e7c73a3c42b5ea014e80bf3fe5a7098ac207673edc4ae34', 'smpl/envs/helpers/constants.py': '9e65f95b8e7cc6d0127df01277e6a6e506978f3f0476b41d0002e0f46df4f935', 'smpl/envs/helpers/helper_funcs.py': 'cca73f5854427aab253ac3c2b4abf985c9b0f24a7e392e33731a666c68c85c33', 'smpl/envs/models/config.py': '412c0b1bd029cdb85ac71a55227f1ddf1041046757144bb13732dcc2c32e9575', 'smpl/configdata/atropineenv/x0.txt': 'e5e6e171e749b0bc763d1b9e939fd0655a151922a8e193f0396cc20445d16c6a', 'smpl/configdata/atropineenv/z0.txt': '6e855e7f095add34ff661a863f63bb7f0920dade25ce870695c9ee7c2f855dcd', 'smpl/configdata/atropineenv/model.npy': 'e02df103c91bb8ca15e5775fb54e98ea5d5f4c0ef1cc7e9943fafdf0329d4f86'}
FLAGS={'t':True,'action':False,'uss':True,'ess':True,'e':True,'kf':True,'z':True,'x':False}


def verify_source(root):
    root=Path(root);actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    if actual!=SOURCE_PINS:raise ValueError('simulator source/model differs from fixed approved B baseline')
    return actual


def experiment_config():
    return {'schema_version':'ProcessRules/1.0','enabled':True,'environment':'Atropine',
        'version':'D-Atropine-isolated-exp-20261116-v1','observation_shape':[39],'action_shape':[4],
        'variables':{name:{'source':'observation','index':index,'unit':'dimensionless','coordinate':'model_observation_normalize_false'}
                     for name,index in [('E_current',6),('E_reference',5)]},
        'rules':[{'rule_id':'R-03-ATROPINE-EXPERIMENT-E','kind':'relative_upper','enabled':True,'priority':1,
            'variable':'E_current','reference_variable':'E_reference','relative_margin':.10,'observable_upper':20.,
            'basis':{'kind':'team_experiment','reference':f'{AUTHORIZATION}:docs/R03_20261116_实验规则确认.md',
                     'reviewed_by':'project user confirmation recorded in PR9; no industrial certification','reviewed_on':'2026-11-16'},
            'suggestion':'Stop new automatic cycles; review cause, require 3 fresh valid states, released resources and human confirmation'}]}


def _fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class ObservedAdapter:
    """Read/execute B via its API; timestamp only actual reset/step returns.

    Unobserved external state revisions invalidate context. Reading cached values
    never manufactures a new timestamp or version. _capture is internal only.
    """
    def __init__(self,adapter,*,fixture=False,clock=time.monotonic,max_age_seconds=2.):
        if type(fixture)is not bool or type(max_age_seconds) not in (int,float) or not np.isfinite(max_age_seconds) or max_age_seconds<=0:
            raise ValueError('explicit fixture flag and finite local age budget required')
        if fixture and type(adapter).__module__!='l3_system.tests.test_live_orchestration_contracts':
            raise ValueError('fixture bypass is restricted to the test-only FakeAdapter')
        self.base=adapter;self.fixture=fixture;self.clock=clock;self.max_age_seconds=max_age_seconds
        self.episode_id=None;self.observed_at=None;self.captured_version=None;self.captured_step=None
        self.captured_observation=None;self.unknown_result=False;self.block_new_steps=lambda:False;self.pre_step_review=lambda action:False
        self.operating_config=self._configuration();self.config_sha256=_fingerprint(self.operating_config)
        self.source_hashes={}
        if not fixture:
            root=Path(adapter.source_dir)
            self.source_hashes=verify_source(root)
    def __getattr__(self,name):return getattr(self.base,name)
    def _configuration(self):
        if tuple(self.base.observation_space.shape)!=(39,) or tuple(self.base.action_space.shape)!=(4,):raise ValueError('Atropine39/4 only')
        if not self.fixture:
            env=getattr(self.base,'_env',None)
            if type(env).__name__!='AtropineEnvGym' or type(env).__module__!='smpl.envs.atropineenv':raise ValueError('verified Atropine backend required; mAb/real device excluded')
            if env.normalize is not False or env.uss_subtracted is not True or env.max_steps!=60:raise ValueError('unapproved operating mode or horizon')
            if any(getattr(env,'observation_include_'+key,None) is not value for key,value in FLAGS.items()):raise ValueError('unapproved observation layout')
        return {'environment':'AtropineEnvGym','fixture':self.fixture,'normalize':False,'uss_subtracted':True,
                'observation_flags':dict(FLAGS),'max_steps':60,'controller':'existing_space_midpoint_interface_baseline',
                'observation_shape':[39],'action_shape':[4],
                'action_low':np.asarray(self.base.action_space.low).tolist(),'action_high':np.asarray(self.base.action_space.high).tolist(),
                'experiment_goal_id':'trigger-gate-takeover-v1','rule_version':'D-Atropine-isolated-exp-20261116-v1','max_local_observation_age_seconds':self.max_age_seconds}
    def _capture(self):
        values=np.asarray(self.base.get_observation());version=self.base.state_version;step=self.base.step_count;now=self.clock()
        if values.shape!=(39,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or type(version)is not int or type(step)is not int or not np.isfinite(now):raise ValueError('invalid B return/context')
        self.captured_observation=values.copy();self.captured_version=version;self.captured_step=step;self.observed_at=now
    def reset(self,seed=1101):
        if self.block_new_steps():raise RuntimeError('reset is not an approved takeover recovery path')
        if not self.fixture and (type(seed)is not int or seed not in (1101,1102,1103)):raise ValueError('seed outside approved first-run baseline')
        result=self.base.reset(seed=seed);self.episode_id=str(uuid4());self.unknown_result=False;self._capture();return result
    def step(self,action):
        if self.block_new_steps():raise RuntimeError('takeover latch forbids automatic or hidden recovery steps')
        expected=space_midpoint(self.base.get_observation(),self.base.action_space)
        if not np.array_equal(np.asarray(action),expected):raise RuntimeError('controller outside approved midpoint experiment')
        if self.pre_step_review(action) is not True:raise RuntimeError('fresh D review required immediately before B step')
        try:
            result=self.base.step(action)
            if result.get('success') is not True:raise RuntimeError('B returned unsuccessful or uncertain execution')
            self._capture();return result
        except Exception:
            self.unknown_result=True;self.observed_at=None
            raise
    def snapshot(self):
        self.base.require_ready()
        if self.unknown_result or self.base.done or self.episode_id is None or self.observed_at is None:raise RuntimeError('unready/terminal/unknown B context')
        if self._configuration()!=self.operating_config:raise RuntimeError('operating configuration changed')
        now=self.clock()
        if not np.isfinite(now) or not 0<=now-self.observed_at<=self.max_age_seconds:raise RuntimeError('observation clock stale or invalid')
        if (self.base.state_version!=self.captured_version or self.base.step_count!=self.captured_step
                or not np.array_equal(np.asarray(self.base.get_observation()),self.captured_observation)):
            raise RuntimeError('B changed without a trusted captured sample')
        return {'episode_id':self.episode_id,'state_version':self.captured_version,'step_count':self.captured_step,
                'observed_at':self.observed_at,'reviewed_at':now,'config_sha256':self.config_sha256,
                'observation':self.captured_observation.tolist()}


class ExperimentMonitor(MonitoringAgent):
    def __init__(self,source,on_reject):
        self.source=source;self.on_reject=on_reject
        super().__init__(source.action_space,source.observation_space,
            process_rules=ProcessRuleEngine(experiment_config(),allow_experimental=True),process_context_provider=self._context)
    def _context(self,state,action):
        sample=self.source.snapshot()
        if not np.array_equal(np.asarray(state),np.asarray(sample['observation'])):raise ValueError('review observation differs from B')
        return {'environment':'Atropine','episode_id':sample['episode_id'],
            'variables':{name:{'unit':'dimensionless','coordinate':'model_observation_normalize_false'} for name in ('E_current','E_reference')}}
    def _remember(self,result):
        if not result['approved']:self.on_reject(result['reason'])
        return result
    def check_action(self,*args,**kwargs):return self._remember(super().check_action(*args,**kwargs))
    def check_observation(self,*args,**kwargs):return self._remember(super().check_observation(*args,**kwargs))
    def check_request(self,request,observation=None):
        result=super().check_request(request,observation)
        if result['decision']=='pass':
            try:
                sample=self.source.snapshot()
                if type(request.get('state_version'))is not int or request['state_version']!=sample['state_version']:raise ValueError('request is stale')
            except Exception:
                result.update(decision='reject',rule_id='R-03-ATROPINE-CONTEXT',reason='Request/live state context mismatch',risk_level=3)
                self.logs[-1].update(approved=False,status='Reject',level=3,rule_id=result['rule_id'],reason=result['reason'],intervention='BLOCK')
                self.on_reject(result['reason'])
        return result


class AtropineExperimentSession:
    def __init__(self,adapter,output_dir,*,enable_approved_experiment=False,fixture=False,clock=time.monotonic):
        if enable_approved_experiment is not True:raise ValueError('explicit approved-isolated-experiment opt-in required')
        if not fixture and hashlib.sha256((Path(__file__).resolve().parents[1]/'runtime.py').read_bytes()).hexdigest()!=APPROVED_A_RUNTIME_SHA256:
            raise ValueError('approved A runtime required; use the combined main/PR workspace, do not silently run an older A baseline')
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=False)
        self.source=ObservedAdapter(adapter,fixture=fixture,clock=clock)
        self.latched=False;self.reason=None;self.recovery=[];self.events=[];self.last_recovery_version=None
        self.source.block_new_steps=lambda:self.latched
        self.monitor=ExperimentMonitor(self.source,self._latch)
        self.source.pre_step_review=lambda action:self.monitor.check_action(self.source.get_observation(),action)['approved']
        self.executor=ExecutionAgent(self.source,self.output_dir/'B-private.jsonl')
        self.system=MultiAgentSystem(executor=self.executor,monitor=self.monitor,event_path=self.output_dir/'events-private.jsonl',max_action_regenerations=0)
    def _event(self,event,**fields):
        record={'timestamp':datetime.now(timezone.utc).isoformat(),'event':event,**fields};self.events.append(record)
        with (self.output_dir/'D-events-private.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    def _latch(self,reason):
        self.latched=True;self.reason=reason;self.recovery=[];self.last_recovery_version=None
        self._event('takeover_latched',reason=reason)
    def start(self,seed=1101):return self.executor.reset(seed=seed)
    def run_cycle(self):
        if self.latched:raise RuntimeError('automatic cycles blocked until full recovery review')
        try:
            result=self.system.run_cycle()
            if self.system.state!='autonomous':self._latch(result['summary']['failure_reason'])
            return result
        except Exception:
            self._latch('cycle/context/unknown execution exception');raise
    def observe_recovery(self):
        if not self.latched:raise RuntimeError('no active takeover')
        try:
            sample=self.source.snapshot()
            if sample['step_count']<=0:raise RuntimeError('reset is not a new recovery observation')
            result=self.monitor.check_observation(sample['observation'])
            if not result['approved']:raise RuntimeError('recovery rule rejected')
        except Exception:
            self.recovery=[];self.last_recovery_version=None;raise RuntimeError('invalid recovery sample; takeover retained') from None
        identity=(sample['episode_id'],sample['config_sha256'])
        if self.recovery and identity!=self.recovery[-1]['identity']:self.recovery=[];self.last_recovery_version=None
        version=sample['state_version']
        if self.last_recovery_version is not None and version<=self.last_recovery_version:
            self._event('recovery_duplicate_ignored',state_version=version)
            return {'fresh_states':len(self.recovery),'new_sample':False,'execution_authorized':False}
        if self.last_recovery_version is not None and version!=self.last_recovery_version+1:self.recovery=[]
        self.last_recovery_version=version;self.recovery.append({'identity':identity,'version':version});self.recovery=self.recovery[-3:]
        self._event('recovery_observation',state_version=version,fresh_states=len(self.recovery))
        return {'fresh_states':len(self.recovery),'new_sample':True,'execution_authorized':False}
    def resume(self,*,operator,reason,cause_reviewed=False):
        if not self.latched or not isinstance(operator,str) or not operator.strip() or not isinstance(reason,str) or not reason.strip() or cause_reviewed is not True:
            raise RuntimeError('takeover, operator, reason and explicit cause review required')
        try:sample=self.source.snapshot()
        except Exception:
            self._latch('invalid context during human recovery review')
            raise RuntimeError('invalid context revoked recovery evidence') from None
        if (len(self.recovery)!=3 or self.recovery[-1]['version']!=sample['state_version']
                or any(r['identity']!=(sample['episode_id'],sample['config_sha256']) for r in self.recovery)):
            raise RuntimeError('three fresh same-context valid states required')
        if self.system.scheduler.active_agent_tasks or self.executor._pending or self.source.unknown_result:
            raise RuntimeError('resources/pending execution/unknown result unresolved')
        self.system.resume(operator=operator,reason=reason,reset=False)
        self._event('human_confirmed_recovery',operator=operator,reason=reason)
        self.latched=False;self.reason=None;self.recovery=[];self.last_recovery_version=None
        return {'state':'autonomous','executed':False}
    def summary(self):
        import sys,importlib.metadata
        versions={'python':sys.version.split()[0]}
        for package in ('numpy','casadi','gym','scipy','requests'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]='unavailable-in-fixture'
        code_hashes={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                     for name in ('runtime.py','main.py','execution/environment_adapter.py','execution/execution_agent.py','monitoring/atropine_experiment.py','agents/process_rules.py')}
        return {'schema_version':'AtropineIsolatedExperiment/1.0','authorization_commit':AUTHORIZATION,'software_versions':versions,'code_file_sha256':code_hashes,
            'experiment_enabled':True,'industrial_safety_certified':False,'fixture':self.source.fixture,
            'config':deepcopy(self.source.operating_config),'config_sha256':self.source.config_sha256,
            'rule_config_sha256':_fingerprint(experiment_config()),'source_file_sha256':self.source.source_hashes,
            'step_calls':self.source.step_calls,'state':'takeover_requested' if self.latched else self.system.state,
            'recovery_fresh_states':len(self.recovery),'automatic_recovery_sampling':False}


if __name__=='__main__':
    import argparse,contextlib,io,os
    from l3_system.execution.environment_adapter import EnvironmentAdapter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-approved-experiment',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=Path(os.environ.get('SMPL_SOURCE_DIR') or Path(__file__).resolve().parents[2]/'vendor/smpl')
    verify_source(source)
    args.output.mkdir(parents=True,exist_ok=False);summaries=[]
    for seed in (1101,1102,1103):
        with EnvironmentAdapter(max_steps=60) as adapter:
            with contextlib.redirect_stdout(io.StringIO()):
                session=AtropineExperimentSession(adapter,args.output/str(seed),enable_approved_experiment=True)
                session.start(seed=seed)
                for _ in range(3):
                    result=session.run_cycle()
                    if session.latched:break
            summary=session.summary();summary['seed']=seed;summaries.append(summary)
    complete=all(e['step_calls']==3 and e['state']=='autonomous' for e in summaries)
    report={'scope':'approved_isolated_Atropine_interface_baseline','episodes':summaries,'baseline_completed':complete,
        'no_physical_anomaly_ground_truth':True,'raw_logs_publication_authorized':False}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))
    raise SystemExit(0 if complete else 1)
