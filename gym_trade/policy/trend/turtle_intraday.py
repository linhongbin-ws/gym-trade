import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from gym_trade.policy.registry import register_policy, register_function
from gym_trade.policy.base import BasePolicy


@register_policy
class Policy(BasePolicy):
    """Turtle-style intraday breakout + pyramid policy.

    Entry: minute close > prior `lookback_days` daily high (column `ta@d52h`)
           AND close >= prev_day_close * (1 + entry_gain_pct) (column `ta@prev_close`).
    Pyramid: each additional tranche fires when close >= last tranche price * (1 + step_pct),
             up to n_max tranches; each tranche is a fixed cash slice = init_balance / n_max.
    Exit:
        - First tranche only: stop at first_price * (1 - initial_stop_pct).
        - 2+ tranches: stop at the previous (second-highest) tranche entry price.
        - After exit, no more entries today (cooldown until EOD, env force-closes).
    """

    def __init__(self, **kwargs):
        hyper_param_range = {
            "step_pct":         (0.05,  0.005, 0.20),
            "n_max":            (5,     1,     20),
            "initial_stop_pct": (0.03,  0.005, 0.20),
            "entry_gain_pct":   (0.10,  0.0,   1.00),   # min gain vs prev day close
            "init_balance":     (1e6,   1e3,   1e9),
        }
        super().__init__(hyper_param_range=hyper_param_range)

    def init_policy(self, **kwargs):
        self._tranche_prices: list[float] = []
        self._target_pos: int = 0
        self._exit_done: bool = False

    def __call__(self, obs, **kwargs):
        close = float(obs["close"])
        d52h = float(obs["ta@d52h"])
        prev_close = float(obs["ta@prev_close"])
        balance = float(obs["dash@balance"])

        n = len(self._tranche_prices)
        n_max = int(self.hyper_param["n_max"])
        step_pct = float(self.hyper_param["step_pct"])
        stop_pct = float(self.hyper_param["initial_stop_pct"])
        gain_pct = float(self.hyper_param["entry_gain_pct"])
        init_balance = float(self.hyper_param["init_balance"])
        tranche_cash = init_balance / n_max

        hold_info = {"entry_point": False, "exit_point": False}

        if (self._exit_done
                or not np.isfinite(d52h) or not np.isfinite(prev_close)
                or not np.isfinite(close) or close <= 0 or prev_close <= 0):
            return np.array([0.0, 0.0]), hold_info

        # ── flat: scan for breakout entry ───────────────────────────────────
        if n == 0:
            gain_ok = (close >= prev_close * (1.0 + gain_pct))
            if close > d52h and gain_ok:
                new_shares = int(tranche_cash / close)
                if new_shares <= 0:
                    return np.array([0.0, 0.0]), hold_info
                self._target_pos = new_shares
                self._tranche_prices.append(close)
                action_1 = min(1.0, self._target_pos * close / balance) if balance > 0 else 0.0
                return np.array([1.0, action_1]), {"entry_point": True, "exit_point": False}
            return np.array([0.0, 0.0]), hold_info

        # ── have position: check stop first ─────────────────────────────────
        if n == 1:
            stop_price = self._tranche_prices[0] * (1.0 - stop_pct)
        else:
            stop_price = self._tranche_prices[-2]   # second-highest entry

        if close <= stop_price:
            self._tranche_prices = []
            self._target_pos = 0
            self._exit_done = True
            return np.array([1.0, 0.0]), {"entry_point": False, "exit_point": True}

        # ── pyramid up ──────────────────────────────────────────────────────
        if n < n_max and close >= self._tranche_prices[-1] * (1.0 + step_pct):
            new_shares = int(tranche_cash / close)
            if new_shares > 0:
                self._target_pos += new_shares
                self._tranche_prices.append(close)
                action_1 = min(1.0, self._target_pos * close / balance) if balance > 0 else 0.0
                return np.array([1.0, action_1]), {"entry_point": True, "exit_point": False}

        return np.array([0.0, 0.0]), hold_info

    @property
    def obs_keys(self):
        return ["close", "ta@d52h", "ta@prev_close", "dash@pos", "dash@balance"]


@register_function
def features(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """No intraday features needed — `ta@d52h` and `ta@prev_close` are injected externally."""
    return pd.DataFrame(index=df.index), {
        "ta@d52h":       [0.0, np.inf],
        "ta@prev_close": [0.0, np.inf],
    }
