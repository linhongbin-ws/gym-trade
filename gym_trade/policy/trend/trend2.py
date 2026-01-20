from __future__ import annotations
from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy

from dataclasses import dataclass
from typing import List, Optional

# Assumes these exist in your codebase:
# - BasePolicy
# - register_policy


@dataclass
class DailyDJIConfig:
    # --- Trend/Momentum thresholds (daily bars) ---
    buy_ema: float = 0.0010        # ema_diff_norm threshold to enter
    sell_ema: float = -0.0005      # ema_diff_norm threshold to exit (hysteresis)

    buy_mom: float = 0.0020        # momentum score threshold to enter
    sell_mom: float = -0.0010      # momentum score threshold to exit (hysteresis)

    # Weighting for momentum score
    w_mom5: float = 0.6
    w_mom15: float = 0.4

    # Optional minimum holding period (in bars/days)
    min_hold_bars: int = 3

    # Optional simple "cooldown" after exiting, to reduce churn
    cooldown_bars: int = 1


@register_policy
class Policy(BasePolicy):
    """
    Discrete daily policy for DJI.

    Action semantics:
      +1 : buy with all cash (go long)
       0 : hold
      -1 : sell all position (go flat)

    Required obs keys (as requested):
      - "dash@pos" (position size; >0 means in position)
      - "rsi_standard@ema_diff_norm"
      - "rsi_standard@mom_5"
      - "rsi_standard@mom_15"
    """

    def __init__(self, obs_keys: List[str], config: Optional[DailyDJIConfig] = None, **kwargs):
        self._obs_keys = obs_keys
        self.cfg = config or DailyDJIConfig()

        # state (bar-count based; your env likely calls policy once per bar)
        self._bars_in_pos = 0
        self._cooldown_left = 0

    def init_policy(self):
        self._bars_in_pos = 0
        self._cooldown_left = 0

    def __call__(self, obs, **kwargs):
        # --- Read required keys exactly as you specified ---
        pos = int(obs["dash@pos"] > 0)
        ema_diff = float(obs["rsi_standard@ema_diff_norm"])
        mom5 = float(obs["rsi_standard@mom_5"])
        mom15 = float(obs["rsi_standard@mom_15"])

        # momentum score (daily)
        mom_score = self.cfg.w_mom5 * mom5 + self.cfg.w_mom15 * mom15

        # update state counters
        if pos == 1:
            self._bars_in_pos += 1
        else:
            self._bars_in_pos = 0
            if self._cooldown_left > 0:
                self._cooldown_left -= 1

        # --- Decision (desired position in {0,1}) ---
        desired = pos

        # If in cooldown, do not re-enter
        if pos == 0 and self._cooldown_left > 0:
            desired = 0
        else:
            if pos == 0:
                # Enter only if BOTH trend and momentum confirm (strong daily filter)
                if (ema_diff > self.cfg.buy_ema) and (mom_score > self.cfg.buy_mom):
                    desired = 1
            else:
                # Exit rules:
                # - Respect min holding period, unless trend+mom strongly flips
                hold_ok = (self._bars_in_pos < self.cfg.min_hold_bars)

                exit_signal = (ema_diff < self.cfg.sell_ema) or (mom_score < self.cfg.sell_mom)

                if exit_signal and not hold_ok:
                    desired = 0

        # --- Map desired -> action ---
        if desired == 1 and pos == 0:
            return 1
        if desired == 0 and pos == 1:
            # start cooldown after exit
            self._cooldown_left = self.cfg.cooldown_bars
            return -1
        return 0

    @property
    def obs_keys(self):
        return self._obs_keys


# # -------------------------
# # Optional: two alternative policies using the same keys
# # (Keep in the same file if you want multiple choices)
# # -------------------------

# @register_policy
# class PolicyMomentumOnly(BasePolicy):
#     """
#     Uses only mom5/mom15 score with hysteresis (still using your exact keys).
#     """
#     def __init__(self, obs_keys: List[str], **kwargs):
#         self._obs_keys = obs_keys
#         self.buy_mom = 0.0020
#         self.sell_mom = -0.0010
#         self.w5 = 0.6
#         self.w15 = 0.4

#     def init_policy(self):
#         pass

#     def __call__(self, obs, **kwargs):
#         pos = int(obs["dash@pos"] > 0)
#         mom5 = float(obs["rsi_standard@mom_5"])
#         mom15 = float(obs["rsi_standard@mom_15"])
#         score = self.w5 * mom5 + self.w15 * mom15

#         desired = pos
#         if pos == 0 and score > self.buy_mom:
#             desired = 1
#         elif pos == 1 and score < self.sell_mom:
#             desired = 0

#         if desired == 1 and pos == 0: return 1
#         if desired == 0 and pos == 1: return -1
#         return 0

#     @property
#     def obs_keys(self):
#         return self._obs_keys


# @register_policy
# class PolicyEMAHysteresisOnly(BasePolicy):
#     """
#     Uses only ema_diff_norm with hysteresis (still using your exact keys).
#     """
#     def __init__(self, obs_keys: List[str], **kwargs):
#         self._obs_keys = obs_keys
#         self.buy_ema = 0.0010
#         self.sell_ema = -0.0005

#     def init_policy(self):
#         pass

#     def __call__(self, obs, **kwargs):
#         pos = int(obs["dash@pos"] > 0)
#         ema_diff = float(obs["rsi_standard@ema_diff_norm"])

#         desired = pos
#         if pos == 0 and ema_diff > self.buy_ema:
#             desired = 1
#         elif pos == 1 and ema_diff < self.sell_ema:
#             desired = 0

#         if desired == 1 and pos == 0: return 1
#         if desired == 0 and pos == 1: return -1
#         return 0

#     @property
#     def obs_keys(self):
#         return self._obs_keys
