from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as np


@register_policy
class Policy(BasePolicy):
    """
    Big buyer behavior classifier:
    - Buy  -> big buyer accumulating
    - Sell -> big buyer distributing (LESS SENSITIVE version)
    - Else -> neutral
    """

    def __init__(self, obs_keys: list[str], **kwargs):

        # NOTE: sell signal thresholds are adjusted to be LESS SENSITIVE
        hyper_param_range = {
            # -------- Accumulation (buying) --------
            "acc_score_buy": (3.0, 1.0, 5.0),
            "price_eff_buy": (0.8, 0.4, 1.2),
            "clv_buy": (0.55, 0.4, 0.8),

            # -------- Distribution (selling)  (LESS SENSITIVE) --------
            # ↓↓↓ tighter conditions (requires stronger evidence)
            "acc_score_sell": (0.8, 0.0, 2.0),    # was 1.5
            "price_eff_sell": (1.4, 1.0, 2.5),    # was 1.2
            "clv_sell": (0.35, 0.2, 0.55),        # was 0.45
        }

        super().__init__(hyper_param_range=hyper_param_range)

    def __call__(self, obs, **kwargs):

        acc_score = float(obs["acc_v2@accumulation_score"])
        price_eff = float(obs["acc_v2@price_efficiency"])
        clv = float(obs["acc_v2@clv_mean"])

        # -------- Big buyer buying (accumulation) --------
        buy_signal = (
            acc_score >= self.hyper_param["acc_score_buy"] and
            price_eff <= self.hyper_param["price_eff_buy"] and
            clv >= self.hyper_param["clv_buy"]
        )

        # -------- Big buyer selling (distribution)
        # -------- LESS SENSITIVE version --------
        sell_signal = (
            acc_score <= self.hyper_param["acc_score_sell"] and
            price_eff >= self.hyper_param["price_eff_sell"] and
            clv <= self.hyper_param["clv_sell"]
        )

        if buy_signal:
            return np.array([1, 1]), {"entry_point": True, "exit_point": False}

        if sell_signal:
            return np.array([1, 0]), {"entry_point": False, "exit_point": True}

        # Neutral
        return np.array([0, 0]), {"entry_point": False, "exit_point": False}


    @property
    def obs_keys(self):
        return [
            "acc_v2@accumulation_score",
            "acc_v2@price_efficiency",
            "acc_v2@clv_mean",
        ]
