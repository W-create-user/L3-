"""Research adapter for joint-action experiments, not an approved MADDPG environment."""
from uuid import uuid4

class JointActionResearchEnv:
    def __init__(self,gateway,*,research_only=False):
        if research_only is not True:raise RuntimeError('C05 unavailable: training is disabled; opt in to research diagnostics only')
        self.gateway=gateway
    def step(self,actions,*,timeout_seconds=5):
        expected=set(self.gateway.partitions)
        if not isinstance(actions,dict) or set(actions)!=expected:raise ValueError('One component for every process actor required')
        cycle='research-'+str(uuid4());context=self.gateway.begin(cycle,timeout_seconds=timeout_seconds)
        for actor,action in actions.items():
            result=self.gateway.submit(cycle,actor,action,state_version=context['state_version'],episode_id=context['episode_id'],coordinate=context['coordinate'])
            if result.get('finalized'):return result
        return self.gateway.finish(cycle)
