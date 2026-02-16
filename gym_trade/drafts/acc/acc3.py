from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as np


@register_policy
class Policy(BasePolicy):
    """
    Institutional accumulation classifier v3

    Buy  -> strong institutional accumulation
    Sell -> strong distribution
    Else -> neutral
    """

    def __init__(self, obs_keys: list[str], **kwargs):

        hyper_param_range = {
            "score_buy": (4.0, 3.0, 6.0),
            "score_sell": (1.5, 0.0, 3.0),

            "clv_buy": (0.55, 0.45, 0.75),
            "price_eff_buy": (0.8, 0.4, 1.2),

            "clv_sell": (0.4, 0.2, 0.6),
            "price_eff_sell": (1.2, 0.9, 2.0),
        }

        super().__init__(hyper_param_range=hyper_param_range)

    def __call__(self, obs, **kwargs):

        score = float(obs["acc_v3@institutional_score"])
        price_eff = float(obs["acc_v3@price_efficiency"])
        clv = float(obs["acc_v3@clv_mean"])

        pos = float(obs["dash@pos"])   # ⭐ 当前仓位

        # ------------------------------------------------
        # signal detection (independent of position)
        # ------------------------------------------------
        buy_signal = (
            score >= self.hyper_param["score_buy"] and
            price_eff <= self.hyper_param["price_eff_buy"] and
            clv >= self.hyper_param["clv_buy"]
        )

        sell_signal = (
            score <= self.hyper_param["score_sell"] and
            price_eff >= self.hyper_param["price_eff_sell"] and
            clv <= self.hyper_param["clv_sell"]
        )

        # ------------------------------------------------
        # position-aware execution logic
        # ------------------------------------------------

        # ===== no position → can open =====
        if pos == 0:
            if buy_signal:
                return np.array([1, 1]), {
                    "entry_point": True,
                    "exit_point": False
                }
            else:
                return np.array([0, 0]), {
                    "entry_point": False,
                    "exit_point": False
                }

        # ===== holding long → can exit =====
        if pos > 0:
            if sell_signal:
                return np.array([1, 0]), {
                    "entry_point": False,
                    "exit_point": True
                }
            else:
                return np.array([0, 0]), {
                    "entry_point": False,
                    "exit_point": False
                }

        # ===== fallback =====
        return np.array([0, 0]), {
            "entry_point": False,
            "exit_point": False
        }


    @property
    def obs_keys(self):
        return [
            "acc_v3@institutional_score",
            "acc_v3@price_efficiency",
            "acc_v3@clv_mean",
            "dash@pos",
        ]
