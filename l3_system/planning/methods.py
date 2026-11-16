"""Finite single-process HTN; effects describe verified software facts only.

No model of unconfirmed upstream/downstream chemistry is invented here.
Plans do not execute B or approve D. Dispatch must recheck live predicates.
"""
from dataclasses import dataclass, field
from uuid import uuid4
from .goals import evaluate_goal


@dataclass
class AtomicTask:
    task_id: str
    capability: str
    dependencies: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()


@dataclass
class GoalPlan:
    method: str
    goal_id: str
    tasks: list[AtomicTask]
    evaluation: dict
    plan_id: str = field(default_factory=lambda: 'PLAN-' + uuid4().hex)
    parent_plan_id: str | None = None

    def validate(self):
        nodes = {t.task_id: t for t in self.tasks}
        if len(nodes) != len(self.tasks) or not nodes:
            raise ValueError('empty plan or duplicate task')
        visiting, visited = set(), set()
        def visit(key):
            if key not in nodes:
                raise ValueError('missing dependency')
            if key in visiting:
                raise ValueError('dependency cycle')
            if key in visited:
                return
            visiting.add(key)
            task = nodes[key]
            if not task.capability or not task.completion_criteria:
                raise ValueError('missing capability or completion criteria')
            for dep in task.dependencies:
                visit(dep)
            visiting.remove(key)
            visited.add(key)
        for key in nodes:
            visit(key)

    def ready_tasks(self, facts, completed=()):
        self.validate()
        if not isinstance(facts, dict) or any(type(v) is not bool for v in facts.values()):
            raise ValueError('facts must contain explicit booleans')
        completed = set(completed)
        if completed - {t.task_id for t in self.tasks}:
            raise ValueError('unknown completed task')
        return [t for t in self.tasks if t.task_id not in completed
                and set(t.dependencies) <= completed
                and all(facts.get(p) is True for p in t.preconditions)]

    def verify_completion(self, task_id, facts, completed=()):
        task = next((t for t in self.ready_tasks(facts, completed) if t.task_id == task_id), None)
        if task is None or not all(facts.get(p) is True for p in task.completion_criteria):
            raise ValueError('task not ready or completion criteria unproven')
        # Only after external evidence satisfies criteria may symbolic effects apply.
        return {effect: True for effect in task.effects}


def plan_goal(goal, registry, observation, history, *, capabilities, parent_plan_id=None):
    evaluation = evaluate_goal(observation, history, goal, registry)
    if evaluation['status'] in ('unevaluable', 'failed'):
        raise ValueError('no legal HTN method: ' + str(evaluation['reason']))
    if evaluation['achieved']:
        method = 'report_achieved'
        definitions = [('verify_goal', 'decision', ('observation_valid',), ('goal_achieved',))]
    else:
        method = 'hold_target' if evaluation['in_range'] else 'adjust_target'
        definitions = [
            ('evaluate_goal', 'decision', ('observation_valid',), ('goal_evaluated',)),
            ('select_hold_policy' if evaluation['in_range'] else 'select_adjust_policy', 'decision', ('goal_evaluated',), ('policy_selected',)),
            ('generate_action_intent', 'decision', ('policy_selected',), ('action_intent_created',)),
            ('safety_review', 'monitoring', ('action_intent_created',), ('safety_passed',)),
            ('execute_approved_action', 'environment_execution', ('safety_passed', 'resource_available'), ('execution_result_received',)),
            ('verify_result', 'decision', ('execution_result_received',), ('result_verified',)),
        ]
    if not isinstance(capabilities, (set, frozenset)):
        raise ValueError('capabilities must be an explicit set')
    if any(capability not in capabilities for _, capability, _, _ in definitions):
        raise ValueError('no matching resource capability')
    tasks = []
    for name, capability, pre, criteria in definitions:
        tasks.append(AtomicTask(name, capability, (tasks[-1].task_id,) if tasks else (), pre, criteria, criteria))
    plan = GoalPlan(method, goal['goal_id'], tasks, evaluation, parent_plan_id=parent_plan_id)
    plan.validate()
    return plan
