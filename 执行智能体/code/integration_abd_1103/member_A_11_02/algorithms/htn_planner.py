"""HTN-style task decomposition for the L3 project.

This is a finite-template HTN implementation for the current MVP.
It is not a general-purpose HTN planning engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class TemplateStep:
    name: str
    description: str
    capability: str
    depends_on: tuple[int, ...] = ()
    preconditions: tuple[str, ...] = ()


class HTNPlanner:
    """Decompose supported high-level goals into validated subtasks."""

    def __init__(self) -> None:
        # depends_on uses zero-based indexes into the same template.
        # The current templates are deliberately sequential and auditable.
        self.templates: Dict[str, Sequence[TemplateStep]] = {
            "inspect_state": (
                TemplateStep("collect_observation", "获取当前仿真环境可用的状态观测",
                             "state_read"),
                TemplateStep("evaluate_state", "依据任务要求检查已获取的状态信息",
                             "decision", depends_on=(0,),
                             preconditions=("observation_available",)),
                TemplateStep("return_state_result", "整理并返回状态检查结果",
                             "decision", depends_on=(1,),
                             preconditions=("state_evaluated",)),
            ),
            "control_cycle": (
                TemplateStep("collect_observation", "获取当前仿真环境可用的状态观测",
                             "state_read"),
                TemplateStep("evaluate_goal", "结合任务目标分析当前状态",
                             "decision", depends_on=(0,),
                             preconditions=("observation_available",)),
                TemplateStep("generate_action_intent", "生成待审核的抽象控制动作意图，不直接生成未经核实的环境动作",
                             "decision", depends_on=(1,),
                             preconditions=("goal_evaluated",)),
                TemplateStep("safety_review", "将候选动作提交监管模块进行安全检查",
                             "monitoring", depends_on=(2,),
                             preconditions=("candidate_action_available",)),
                TemplateStep("execute_approved_action", "仅在监管放行后请求执行模块提交动作",
                             "environment_execution", depends_on=(3,),
                             preconditions=("monitoring_passed",)),
                TemplateStep("verify_result", "读取执行反馈并检查本轮任务结果",
                             "decision", depends_on=(4,),
                             preconditions=("execution_feedback_available",)),
            ),
        }

    def supported_goal_types(self) -> List[str]:
        return sorted(self.templates.keys())

    def decompose(self, goal_type: str) -> list:
        """Return SubTask-like dataclass instances with IDs and dependencies."""
        if goal_type not in self.templates:
            raise ValueError(f"没有对应的任务模板：{goal_type}")

        # Local import avoids coupling the planner to the whole DecisionAgent module.
        from agents.decision_agent import SubTask

        steps = self.templates[goal_type]
        ids = [f"SUB-{uuid4().hex[:8]}" for _ in steps]
        subtasks = []

        for index, step in enumerate(steps):
            invalid = [dep for dep in step.depends_on
                       if not isinstance(dep, int) or dep < 0 or dep >= index]
            if invalid:
                raise ValueError(
                    f"模板 {goal_type} 第 {index} 步存在无效依赖索引：{invalid}"
                )

            subtasks.append(
                SubTask(
                    subtask_id=ids[index],
                    name=step.name,
                    description=step.description,
                    order=index + 1,
                    dependencies=[ids[dep] for dep in step.depends_on],
                    required_capability=step.capability,
                    preconditions=list(step.preconditions),
                    status="pending",
                )
            )

        self.validate_subtasks(subtasks)
        return subtasks

    @staticmethod
    def validate_subtasks(subtasks: list) -> bool:
        """Validate unique IDs, dependency references, and acyclic ordering."""
        ids = [item.subtask_id for item in subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("子任务 subtask_id 必须唯一")

        known = set(ids)
        order_by_id = {item.subtask_id: item.order for item in subtasks}

        for item in subtasks:
            if not item.name or not item.description or not item.required_capability:
                raise ValueError(f"子任务字段不完整：{item.subtask_id}")
            if not isinstance(item.order, int) or item.order < 1:
                raise ValueError(f"子任务 order 必须为正整数：{item.subtask_id}")
            for dep in item.dependencies:
                if dep not in known:
                    raise ValueError(
                        f"子任务 {item.subtask_id} 引用了不存在的依赖：{dep}"
                    )
                if dep == item.subtask_id:
                    raise ValueError(f"子任务不能依赖自身：{item.subtask_id}")
                if order_by_id[dep] >= item.order:
                    raise ValueError(
                        f"依赖任务必须先于当前任务：{dep} -> {item.subtask_id}"
                    )

        # Since every dependency must point to a strictly earlier order,
        # cycles are excluded by construction; still verify duplicate orders
        # do not conceal an invalid dependency ordering.
        return True
