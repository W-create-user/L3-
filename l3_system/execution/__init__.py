"""B's SMPL execution interface v1.2; accepts A's raw or mapped Task."""
from .environment_adapter import EnvironmentAdapter
from .execution_agent import ExecutionAgent
from .task_contract import normalize_task

__all__ = ['EnvironmentAdapter', 'ExecutionAgent', 'normalize_task']
