"""Persistent local four-agent session with bounded recovery and human handover.

This is a simulation state machine, not an industrial safety controller.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.decision_agent import DecisionAgent


class MultiAgentSystem:
    def __init__(self, *, executor, monitor, scheduler=None, decision=None,
                 event_path=None, max_action_regenerations=0):
        if type(max_action_regenerations) is not int or max_action_regenerations < 0:
            raise ValueError('max_action_regenerations must be a nonnegative integer')
        if getattr(monitor, 'demo_mode', True):
            raise ValueError('A live-space monitor is required')
        self.executor = executor
        self.monitor = monitor
        self.scheduler = scheduler if scheduler is not None else SchedulingAgent()
        self.decision = decision if decision is not None else DecisionAgent()
        self.event_path = Path(event_path) if event_path else None
        self.max_action_regenerations = max_action_regenerations
        self.state = 'autonomous'
        self.transitions = []
        self.history = []

    def _transition(self, state, reason, **context):
        record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                  'event': 'system_state_changed', 'previous_state': self.state,
                  'state': state, 'reason': reason, **context}
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.transitions.append(record)
        self.state = state

    def run_cycle(self, goal='执行一次仿真控制任务', *, action_generator=None):
        from l3_system.main import run_control_cycle
        if self.state != 'autonomous':
            raise RuntimeError('Human confirmation is required before another cycle')
        attempts = []
        try:
            for attempt in range(self.max_action_regenerations + 1):
                generator = (lambda obs, space, unused: action_generator(obs, space, attempt)) if action_generator else None
                result = run_control_cycle(
                    goal=goal, executor=self.executor, monitor=self.monitor,
                    scheduler=self.scheduler, decision=self.decision,
                    action_generator=generator, event_path=self.event_path)
                attempts.append(result)
                self.history.append(result)
                summary = result['summary']
                if summary['run_status'] == 'completed':
                    if self.state == 'restricted':
                        self._transition('autonomous', 'reviewed_replacement_completed', task_id=summary['task_id'])
                    break
                # Only an explicitly rejected, unexecuted candidate can be replaced.
                # Never repeat an uncertain/failed physical step or checker error.
                if (summary['failure_reason'] == 'safety_rejected'
                        and summary['step_delta'] == 0 and action_generator is not None
                        and attempt < self.max_action_regenerations):
                    self._transition('restricted', 'regenerate_rejected_candidate',
                                     task_id=summary['task_id'], attempt=attempt + 1)
                    continue
                self._transition('takeover_requested', summary['failure_reason'], task_id=summary['task_id'])
                break
        except Exception as exc:
            self._transition('takeover_requested', 'orchestration_exception', error_type=type(exc).__name__)
            raise
        result['summary']['system_state'] = self.state
        result['summary']['action_regenerations'] = len(attempts) - 1
        # Avoid a self-referential result; retain earlier failed plans as evidence.
        result['recovery_attempts'] = attempts[:-1]
        return result

    def resume(self, *, operator, reason, reset=False, seed=1101):
        """Explicit simulated human acknowledgement; never executes a control action.

        reset=True starts a NEW simulation episode and is recorded as such.
        Acknowledgement alone cannot make an invalid/unready environment ready.
        """
        if self.state != 'takeover_requested':
            raise RuntimeError('No pending takeover request')
        if any(not isinstance(x, str) or not x.strip() for x in (operator, reason)):
            raise ValueError('operator and reason must be non-empty strings')
        if type(reset) is not bool:
            raise ValueError('reset must be boolean')
        if self.scheduler.active_agent_tasks:
            raise RuntimeError('Outstanding assignments must be resolved before resuming')
        if reset:
            self._transition('takeover_requested', 'human_requested_new_episode', operator=operator, note=reason)
            self.executor.reset(seed=seed)
        self.executor.adapter.require_ready()
        if self.executor.adapter.done:
            raise RuntimeError('Episode ended; explicitly reset to begin a new episode')
        review = self.monitor.check_observation(self.executor.get_observation())
        if not review['approved']:
            raise RuntimeError('Observation validation failed; takeover remains active')
        self._transition('autonomous', 'human_confirmed_resume', operator=operator,
                         note=reason, new_episode=reset)
        return {'system_state': self.state, 'new_episode': reset}
