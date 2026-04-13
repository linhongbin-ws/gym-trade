import numpy as np
from gym_trade.policy.registry import register_policy, register_function
from gym_trade.policy.base import BasePolicy
import pandas as pd
from typing import Tuple, Dict, List


@register_policy
class Policy(BasePolicy):
    """Buy and Hold: buy full position on the first tick, hold until the end."""

    def __init__(self, **kwargs):
        super().__init__(hyper_param_range={})
        self._entered = False

    def init_policy(self, **kwargs):
        self._entered = False

    def __call__(self, obs, **kwargs):
        if not self._entered:
            self._entered = True
            return np.array([1.0, 1.0]), {"entry_point": True, "exit_point": False}
        return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

    @property
    def obs_keys(self):
        return ["dash@pos", "close"]


@register_function
def features(
    df: pd.DataFrame,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """BNH needs no technical features — return an empty DataFrame."""
    return pd.DataFrame(index=df.index), {}
