"""
Direction-Toggle Strong-Up Trend Policy (long-only, daily)

Source signal: gym_trade.tool.ta.direction_toggle — detects price-direction reversals
and classifies each toggle's surrounding 4-bar shape into pattern_id 1..10. Pattern 1
is "strong up"; the accumulator counts consecutive strong-up patterns and resets on
patterns 6/7/8 (down regimes).

Entry  (flat, signal hot):
    sig_cnt >= sig_cnt_thres            # consecutive bars where strongup_acc >= 1
    pos_ratio < entry_pos_thres         # not already near-full
    → buy to full target (action=[1,1])

Exit   (full, signal cold):
    sig_cnt <  sig_cnt_thres            # accumulator dropped to 0
    pos_ratio > exit_pos_thres
    → flatten (action=[1,0])

NOTE: direction_toggle() in tool/ta.py is O(N²) — fine for daily (~1000 bars), slow
on minute data. If you want to backtest 1m bars on this signal, rewrite that function
to avoid the per-bar dropna scan first.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy, register_function
from gym_trade.tool.ta import direction_toggle


@register_policy
class Policy(BasePolicy):
    def __init__(self, **kwargs):
        hyper_param_range = {
            "sig_cnt_thres":   (1,    1,    10),    # consecutive strongup-acc bars to fire entry
            "entry_pos_thres": (0.9,  0.1,  1.0),   # pos_ratio must be below this to enter
            "exit_pos_thres":  (0.9,  0.1,  1.0),   # pos_ratio must be above this to exit
        }
        super().__init__(hyper_param_range=hyper_param_range)

    def init_policy(self, **kwargs):
        self._sig_cnt = 0

    def __call__(self, obs, **kwargs):
        acc     = float(obs["ta@dt_strongup_acc"])
        pos     = float(obs["dash@pos"])
        balance = float(obs["dash@balance"])
        close   = float(obs["close"])

        if acc >= 1:
            self._sig_cnt += 1
        else:
            self._sig_cnt = 0

        pos_ratio = (pos * close) / balance if balance > 0 else 0.0

        thres       = int(self.hyper_param["sig_cnt_thres"])
        entry_thres = float(self.hyper_param["entry_pos_thres"])
        exit_thres  = float(self.hyper_param["exit_pos_thres"])

        if pos_ratio < entry_thres and self._sig_cnt >= thres:
            return np.array([1.0, 1.0]), {"entry_point": True, "exit_point": False}
        if pos_ratio > exit_thres and self._sig_cnt < thres:
            return np.array([1.0, 0.0]), {"entry_point": False, "exit_point": True}
        return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

    @property
    def obs_keys(self):
        return ["close", "dash@pos", "dash@balance", "ta@dt_strongup_acc"]


@register_function
def features(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Wrap direction_toggle() and re-prefix its outputs under ta@ to match project convention.

    Returns 3 columns the policy or vis layer may consume:
        ta@dt_strongup_acc — consecutive strong-up pattern counter (the entry trigger)
        ta@dt_pattern_id   — 0..10 swing-pattern classification (useful for vis subcharts)
        ta@dt_toggle       — 0/1 flag for direction-reversal bars
    """
    raw = direction_toggle(df, key="close")
    out = pd.DataFrame(
        {
            "ta@dt_strongup_acc": raw["direction_toggle_pattern_strongup_acc@close"],
            "ta@dt_pattern_id":   raw["direction_toggle_pattern_id@close"],
            "ta@dt_toggle":       raw["direction_toggle_bool@close"],
        },
        index=df.index,
    )
    col_range = {
        "ta@dt_strongup_acc": [0.0, np.inf],
        "ta@dt_pattern_id":   [0.0, 10.0],
        "ta@dt_toggle":       [0.0, 1.0],
    }
    return out, col_range
