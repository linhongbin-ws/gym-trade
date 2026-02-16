from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy


@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str], **kwargs ): 
        self._obs_keys = obs_keys 
        self.hyper_param = {}
    def init_policy(self, hyper_search: str | None = None,**kwargs):
        pass
    def __call__(self, obs, **kwargs):
        action = 1
        return action   
    
    @property
    def obs_keys(self): 
        return self._obs_keys