"""Explicit opt-in research hybrid: short source upstream MPC + source buffer P/switch."""
import numpy as np

class SourceMPCPolicy:
    def __init__(self,adapter,*,horizon=2,allow_research=False):
        if allow_research is not True:raise RuntimeError('Source MPC is disabled unless explicitly opted in for research')
        if adapter.metadata.get('environment_id')!='MAbEnvGym':raise ValueError('mAb backend required')
        if type(horizon)is not int or not 1<=horizon<=10:raise ValueError('Bounded research MPC horizon required')
        self.adapter=adapter;self.horizon=horizon;self.solver=None;self._pending=False
        self.solver_statuses=[];self.saturation_count=0;self._pending_saturation=False
    def reset(self):
        c=self.adapter._env.controller
        self.solver=c._build_mpc_up(np.eye(17),np.eye(7),self.horizon,c.dt_spl,c.xss[:17],c.uss[:7],c.xscale[:17],c.uscale[:7])
        self.solver.initialize(casadioptions={'ipopt.linear_solver':'mumps','ipopt.print_level':0,'ipopt.max_iter':100,'ipopt.max_cpu_time':5.},solveroptions={'tol':1e-5})
        self._pending=False;self.solver_statuses=[];self.saturation_count=0
    def predict(self,observation,action_space):
        if self.solver is None or self._pending:raise RuntimeError('Reset policy or commit preceding candidate first')
        raw=np.asarray(observation)
        if raw.shape!=(1970,) or not np.isfinite(raw).all():raise ValueError('Finite model observation required')
        if (tuple(action_space.shape)!=(9,) or not np.array_equal(action_space.low,self.adapter.action_space.low)
                or not np.array_equal(action_space.high,self.adapter.action_space.high)):raise ValueError('MPC action-space mismatch')
        self.solver.fixvar('x',0,raw[:17]);self.solver.solve()
        status=str(self.solver.stats.get('status'));self.solver_statuses.append(status)
        if status!='Solve_Succeeded':raise RuntimeError('MPC solve failed; candidate action is not released: '+status)
        upstream=np.squeeze(self.solver.var['u',0])
        if upstream.shape!=(7,) or not np.isfinite(upstream).all():raise RuntimeError('MPC returned invalid action')
        action=self.adapter._env.steady_actions.copy();action[:7]=upstream
        c=self.adapter._env.controller;drain=c._pcontroller(raw[17])
        bounded_drain=float(np.clip(drain,action_space.low[7],action_space.high[7]));action[7]=bounded_drain
        action[8]=c._switcher(raw[19:])
        action=self.adapter.validate_action(action.astype(action_space.dtype))
        self._pending=True;self._pending_saturation=bool(bounded_drain!=float(drain))
        return action
    def commit(self,executed):
        if type(executed)is not bool or not self._pending:raise ValueError('Pending MPC candidate and boolean execution result required')
        if executed:
            self.solver.saveguess();self.saturation_count+=int(self._pending_saturation)
        self._pending=False
