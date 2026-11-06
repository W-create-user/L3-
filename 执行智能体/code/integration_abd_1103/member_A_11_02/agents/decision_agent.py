"""DecisionAgent: goal recognition, finite-template selection and task creation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Union
from uuid import uuid4
import json

from algorithms.htn_planner import HTNPlanner


@dataclass
class SubTask:
    subtask_id: str
    name: str
    description: str
    order: int
    dependencies: list[str]
    required_capability: str
    preconditions: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class Task:
    task_id: str
    task_name: str
    goal_type: str
    description: str
    priority: int
    preconditions: list[str]
    constraints: Dict[str, Any]
    subtasks: list[SubTask]
    status: str = "created"
    source: str = "rule_mapping"


class DecisionAgent:
    """Create a structured, finite-template task plan from a supported goal."""

    def __init__(self, planner: HTNPlanner | None = None) -> None:
        self.goal_mapping = {
            "检查当前仿真状态": "inspect_state",
            "检查状态": "inspect_state",
            "执行一次仿真控制任务": "control_cycle",
            "执行一次控制任务": "control_cycle",
        }
        self.planner = planner or HTNPlanner()

    def recognize_goal(self, goal: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(goal, str):
            normalized = goal.strip()
            if not normalized:
                raise ValueError("目标字符串不能为空")
            if normalized not in self.goal_mapping:
                raise ValueError(f"暂不支持该目标：{normalized}")
            return {
                "goal_type": self.goal_mapping[normalized],
                "description": normalized,
                "priority": 3,
                "constraints": {},
                "preconditions": [],
            }

        if not isinstance(goal, dict):
            raise TypeError("输入必须是字符串或字典")

        goal_type = goal.get("goal_type")
        if goal_type not in self.planner.supported_goal_types():
            raise ValueError(f"不支持的任务类型：{goal_type}")

        description = goal.get("description", goal_type)
        priority = goal.get("priority", 3)
        constraints = goal.get("constraints", {})
        preconditions = goal.get("preconditions", [])

        if not isinstance(description, str) or not description.strip():
            raise ValueError("description 必须是非空字符串")
        if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
            raise ValueError("priority 必须是 1 到 5 之间的整数")
        if not isinstance(constraints, dict):
            raise ValueError("constraints 必须是字典")
        if not isinstance(preconditions, list) or not all(
            isinstance(item, str) and item.strip() for item in preconditions
        ):
            raise ValueError("preconditions 必须是非空字符串组成的列表")

        # Ensure constraints can cross a JSON-based module boundary.
        try:
            json.dumps(constraints, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("constraints 必须能够序列化为 JSON") from exc

        return {
            "goal_type": goal_type,
            "description": description.strip(),
            "priority": priority,
            "constraints": constraints.copy(),
            "preconditions": [item.strip() for item in preconditions],
        }

    def create_task(self, goal: Union[str, Dict[str, Any]]) -> Task:
        goal_info = self.recognize_goal(goal)
        subtasks = self.planner.decompose(goal_info["goal_type"])

        task = Task(
            task_id=f"TASK-{uuid4().hex[:8]}",
            task_name=goal_info["description"],
            goal_type=goal_info["goal_type"],
            description=goal_info["description"],
            priority=goal_info["priority"],
            preconditions=goal_info["preconditions"],
            constraints=goal_info["constraints"],
            subtasks=subtasks,
        )
        self.validate_task(task)
        return task

    @staticmethod
    def validate_task(task: Task) -> bool:
        if not task.task_id or not task.task_name:
            raise ValueError("总任务必须包含 task_id 和 task_name")
        if not 1 <= task.priority <= 5:
            raise ValueError("任务优先级必须在 1 到 5 之间")
        if not task.subtasks:
            raise ValueError("任务至少包含一个子任务")
        HTNPlanner.validate_subtasks(task.subtasks)
        return True

    @staticmethod
    def task_to_dict(task: Task) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary for other Agent modules."""
        return asdict(task)
