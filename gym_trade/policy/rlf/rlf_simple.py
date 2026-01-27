from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as  np
import gym

@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str],  **kwargs ): 
        
        hyper_param_range = {
            "slope_pct_annual_thres": (0.1, 0.001, 1),
            # "r2_thres": (0.4, 0.001, 1),
            # "t_thres": (2.0, 0.001, 1),
        }
        super().__init__(hyper_param_range=hyper_param_range)


    def __call__(self, obs, **kwargs):
        prefix = "rlf_ma240_60@"
        # trend_up = (
        #     (obs[prefix + "slope_pct_annual"] > self.hyper_param["slope_pct_annual_thres"]) &   
        #     (obs[prefix + "r2"] > self.hyper_param["r2_thres"]) &
        #     (np.abs(obs[prefix + "t"]) > self.hyper_param["t_thres"])
        # )

        # # trend_down = (
        # #     (obs[prefix + "slope_pct_annual"] < -0.05) &
        # #     (obs[prefix + "r2"] > 0.4) &
        # #     (obs[prefix + "t"].abs() > 2.0)
        # # )

        # accel_up = trend_up & (obs[prefix + "slope"] > 0)
        # decel_up = trend_up & (obs[prefix + "slope"] < 0)

        accel_up = obs[prefix + "slope_pct_annual"] > self.hyper_param["slope_pct_annual_thres"]
        decel_up = obs[prefix + "slope_pct_annual"] <= self.hyper_param["slope_pct_annual_thres"]

        if accel_up and obs["dash@pos"] == 0:
            action = 1
        elif decel_up and obs["dash@pos"] > 0:
            action = -1
        else:
            action = 0   
        return action
    @property
    def obs_keys(self): 
        prefix = "rlf_ma240_60@"
        keys = [prefix + k for k in ["slope", "intercept", "r2", "t", "slope_pct_annual"]]
        keys += ["dash@pos"]
        return keys


