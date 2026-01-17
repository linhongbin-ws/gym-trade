from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy

@register_policy
class Policy(BasePolicy):
    def init_policy(self, **kwargs):
        pass
    def __call__(self, obs, **kwargs):
        action = 1
        return action 