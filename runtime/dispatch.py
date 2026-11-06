"""Multiple eligible workers, real CNP bids and bounded reassignment."""
from dataclasses import dataclass
from agents.scheduling_agent import SchedulingAgent


@dataclass
class Worker:
    agent_id: str
    capabilities: set
    cost: float = 1.0
    load: float = 0.0


class Dispatcher:
    def __init__(self, workers):
        self.workers = list(workers)
        if len({w.agent_id for w in self.workers}) != len(self.workers):
            raise ValueError('Worker identifiers must be unique')
        self.scheduler = SchedulingAgent()

    def execute(self, tasks, capabilities, handler, *, strategy='cnp', max_retries=1):
        if strategy not in {'cnp', 'fixed'} or type(max_retries) is not int or max_retries < 0:
            raise ValueError('Invalid dispatch configuration')
        self.scheduler.register_tasks(tasks)
        events = []
        while ready := self.scheduler.ready_tasks():
            for task in ready:
                key = task['subtask_id']
                candidates = [w for w in self.workers if capabilities[key] in w.capabilities]
                bids = [{'task_id': task['task_id'], 'subtask_id': key, 'agent_id': w.agent_id,
                         'estimated_cost': w.cost, 'capability_match': 1., 'load': w.load}
                        for w in candidates]
                if strategy == 'fixed':
                    bids = bids[:1]
                award = self.scheduler.schedule(task, bids)
                for attempt in range(max_retries + 1):
                    if award['assignment_status'] != 'awarded':
                        return {'success': False, 'events': events, 'reason': 'no_eligible_worker'}
                    try:
                        success = handler(award['winner_id'], dict(task)) is True
                        reason = None if success else 'worker_failure'
                    except Exception as exc:
                        success, reason = False, f'{type(exc).__name__}: {exc}'
                    feedback = self.scheduler.update_feedback({
                        'task_id': task['task_id'], 'subtask_id': key,
                        'execution_status': 'completed' if success else 'failed',
                        'failure_reason': reason})
                    events.append({'assignment': award, 'feedback': feedback, 'attempt': attempt})
                    if success:
                        break
                    if attempt == max_retries:
                        return {'success': False, 'events': events, 'reason': reason}
                    award = self.scheduler.retry_failed_task(task['task_id'], key, bids)
        success = all(t['status'] == 'completed' for t in self.scheduler.tasks.values())
        return {'success': success, 'events': events,
                'reason': None if success else 'unresolved_dependencies'}
