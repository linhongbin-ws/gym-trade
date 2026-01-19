from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy


@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str]): 
        self._obs_keys = obs_keys 

    def init_policy(self,):
        pass
    def __call__(self, obs, **kwargs):
        action = 1
        return action   
    
    @property
    def obs_keys(self): 
        return self._obs_keys