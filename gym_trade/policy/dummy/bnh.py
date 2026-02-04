from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as  np

@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str],  **kwargs ): 
        
        hyper_param_range = {
        }
        super().__init__(hyper_param_range=hyper_param_range)


    def __call__(self, obs, **kwargs):
        action = np.zeros(2)
        if obs["dash@pos"] == 0:
            action[0] = 1
            action[1] = 1
        else:
            action[0] = 0 
        return action
    @property
    def obs_keys(self): 
        keys = []
        keys += ["dash@pos"]
        return keys


