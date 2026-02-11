from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as np


@register_policy
class Policy(BasePolicy):
    """
    Trade purely based on accumulation (big buyer behavior),
    no trend / linreg / MA dependency.
    """

    def __init__(self, obs_keys: list[str], **kwargs):

        hyper_param_range = {
            # ---------- Accumulation strength ----------
            "acc_score_enter": (3.0, 1.0, 5.0),
            "acc_score_exit": (1.5, 0.0, 4.0),

            "wick_thres": (0.35, 0.1, 0.9),
            "vol_ratio_enter": (1.15, 0.8, 3.0),
            "vol_ratio_exit": (0.95, 0.5, 2.0),

            # ---------- Behavior control ----------
            "min_hold_steps": (30, 0, 300),
            "cooldown_steps": (5, 0, 200),
            "exit_confirm_steps": (3, 1, 20),
        }

  

        super().__init__(hyper_param_range=hyper_param_range)

        self._hold_steps = 0
        self._cooldown = 0
        self._exit_count = 0

    def __call__(self, obs, **kwargs):
        pos = obs["dash@pos"]

        # -------- Accumulation features --------
        acc_score = float(obs["acc@accumulation_score"])
        wick_ratio = float(obs["acc@lower_wick_ratio"])
        vol_ratio = float(obs["acc@up_down_vol_ratio"])

        # -------- Update state --------
        if pos > 0:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        if self._cooldown > 0:
            self._cooldown -= 1

        # -------- Structure conditions --------
        structure_strong = (
            acc_score >= self.hyper_param["acc_score_enter"] and
            wick_ratio >= self.hyper_param["wick_thres"] and
            vol_ratio >= self.hyper_param["vol_ratio_enter"]
        )

        structure_weak = (
            acc_score <= self.hyper_param["acc_score_exit"] or
            vol_ratio <= self.hyper_param["vol_ratio_exit"]
        )

        # -------- Entry --------
        can_enter = (self._cooldown == 0)
        enter_signal = can_enter and structure_strong

        # -------- Exit --------
        exit_raw = structure_weak

        # -------- Decision --------
        if pos == 0:
            self._exit_count = 0
            if enter_signal:
                return np.array([1, 1]), {"entry_point": True, "exit_point": False}
            return np.array([0, 0]), {"entry_point": False, "exit_point": False}

        # --- Holding ---
        if self._hold_steps < int(self.hyper_param["min_hold_steps"]):
            self._exit_count = 0
            return np.array([0, 0]), {"entry_point": False, "exit_point": False}

        if exit_raw:
            self._exit_count += 1
        else:
            self._exit_count = 0

        if self._exit_count >= int(self.hyper_param["exit_confirm_steps"]):
            self._cooldown = int(self.hyper_param["cooldown_steps"])
            self._exit_count = 0
            return np.array([1, 0]), {"entry_point": False, "exit_point": True}

        return np.array([0, 0]), {"entry_point": False, "exit_point": False}

    @property
    def obs_keys(self):
        return [
            "acc@accumulation_score",
            "acc@lower_wick_ratio",
            "acc@up_down_vol_ratio",
            "dash@pos",
        ]
