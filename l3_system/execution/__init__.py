"""B's SMPL execution interface v1.2; accepts A's raw or mapped Task."""
from .environment_adapter import EnvironmentAdapter
from .execution_agent import ExecutionAgent
from .task_contract import normalize_task

__all__ = ['EnvironmentAdapter', 'ExecutionAgent', 'normalize_task']

from .base_adapter import BaseAdapter
from .mab_adapter import MAbAdapter

def create_adapter(environment_id, **kwargs):
    """Select explicitly; no backend inference from dimensions or dependency presence."""
    factories={'AtropineEnvGym':EnvironmentAdapter,'MAbEnvGym':MAbAdapter}
    if environment_id not in factories:raise ValueError('Unknown environment_id')
    return factories[environment_id](**kwargs)

__all__ += ['BaseAdapter','MAbAdapter','create_adapter']
