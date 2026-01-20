from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy


@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str], **kwargs):
        self._obs_keys = obs_keys
        self.buy_ema = 0.0005
        self.sell_ema = -0.0002
        self.buy_mom = 0.0010
        self.sell_mom = -0.0005

    def init_policy(self): 
        pass
    def __call__(self, obs, **kwargs):
        for k in self._obs_keys:
            assert k in obs, f"obs_keys {self._obs_keys} not in obs {obs.keys()}"

        pos = int(obs["dash@pos"] > 0)
        ema_diff = obs["rsi_standard@ema_diff_norm"]
        mom5 = obs["rsi_standard@mom_5"]
        mom15 = obs["rsi_standard@mom_15"]

        score = 0.6 * mom5 + 0.4 * mom15

        desired = pos
        if pos == 0:
            if (ema_diff > self.buy_ema) and (score > self.buy_mom):
                desired = 1
        else:
            if (ema_diff < self.sell_ema) or (score < self.sell_mom):
                desired = 0

        if desired == 1 and pos == 0: return 1
        if desired == 0 and pos == 1: return -1
        return 0

    @property
    def obs_keys(self):
        return self._obs_keys
