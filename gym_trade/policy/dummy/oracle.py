from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as  np
from typing import List
from datetime import datetime
import pandas as pd

@register_policy
class Policy(BasePolicy):
    def __init__(self, buy_dates: List[str], sell_dates: List[str], **kwargs): 
        
        hyper_param_range = {
        }
        super().__init__(hyper_param_range=hyper_param_range)
        self._buy_dates = set(pd.to_datetime(buy_dates).normalize())
        self._sell_dates = set(pd.to_datetime(sell_dates).normalize())
        # print("buy dates", self._buy_dates)
        # print("sell dates", self._sell_dates)
    def __call__(self, obs, **kwargs):
        obs_date = pd.Timestamp(obs["index_datetime"]).normalize()

        if obs_date in self._buy_dates and obs["dash@pos"] == 0:
            action = 1
            print("action buy date", obs_date)
        elif obs_date in self._sell_dates and obs["dash@pos"] > 0:
            action = -1
            print("action sell date", obs_date)
        else:
            action = 0
        return action
    @property
    def obs_keys(self): 
        keys = []
        keys += ["dash@pos", "index_datetime"]
        return keys


