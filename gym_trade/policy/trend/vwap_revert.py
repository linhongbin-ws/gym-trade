"""
VWAP Mean-Reversion Strategy (long-only, intraday 1m)

Universe (pre-filtered externally):
  prev_close > $5, ADDV20 > $10M

In-play filter (applied each bar):
  09:45 ≤ time ≤ 15:30
  rel_vol       > 1.5
  |ret_since_open| > 1%
  intraday_range_so_far > 1.5%

Long entry:
  vwap_dev  < -0.6%   (price stretched below VWAP)
  z20       < -1.5    (statistically oversold vs 20-bar mean)
  ret_5m    > -2.5%   (not in free-fall)

Exit (first condition that fires):
  |vwap_dev| < vwap_revert_thres  (price reverted to VWAP)
  hold_steps >= max_hold_steps    (15-minute time stop)
  pnl        <= -stop_loss        (stop loss -0.35%)
  pnl        >= take_profit       (take profit +0.5%)
  tod        >= 945               (15:45 force close)

Note: short entries require a short-capable env; current env enforces pos >= 0,
so only the long side is active.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from gym_trade.policy.registry import register_policy, register_function
from gym_trade.policy.base import BasePolicy


# ──────────────────────────────────────────────
#  Policy
# ──────────────────────────────────────────────

@register_policy
class Policy(BasePolicy):
    def __init__(self, **kwargs):
        hyper_param_range = {
            # ── in-play filter ──────────────────────────────
            "rel_vol_thres":      (1.50,  0.50,  5.00),   # rel_volume > X
            "ret_open_thres":     (0.010, 0.003, 0.050),  # |ret_since_open| > X
            "range_thres":        (0.015, 0.005, 0.050),  # intraday_range_pct > X

            # ── entry ───────────────────────────────────────
            "vwap_dev_thres":     (0.006, 0.001, 0.030),  # |vwap_dev| > X  (0.6%)
            "z20_thres":          (1.50,  0.30,  4.00),   # |z20| > X
            "ret_5m_thres":       (0.025, 0.003, 0.100),  # |ret_5m| < X  (not in freefall)

            # ── exit ────────────────────────────────────────
            "vwap_revert_thres":  (0.002, 0.000, 0.010),  # |vwap_dev| < X → reverted
            "max_hold_steps":     (15,    3,     60),      # bars to time-out
            "stop_loss":          (0.0035, 0.001, 0.020),  # -0.35%
            "take_profit":        (0.0050, 0.002, 0.030),  # +0.50%
            "force_close_tod":    (945,   900,   960),     # 15:45 → tod in minutes
        }
        super().__init__(hyper_param_range=hyper_param_range)

        self._ta = "ta@"
        # per-episode state (reset by init_policy)
        self._hold_steps  = 0
        self._entry_price = None

    def init_policy(self, **kwargs):
        self._hold_steps  = 0
        self._entry_price = None

    # ------------------------------------------------------------------
    def __call__(self, obs, **kwargs):
        pos   = float(obs["dash@pos"])
        close = float(obs["close"])
        tod   = float(obs[self._ta + "tod"])

        vwap_dev  = float(obs[self._ta + "vwap_dev"])
        z20       = float(obs[self._ta + "z20"])
        rel_vol   = float(obs[self._ta + "rel_vol"])
        ret_open  = float(obs[self._ta + "ret_since_open"])
        range_pct = float(obs[self._ta + "range_pct"])
        ret_5m    = float(obs[self._ta + "ret_5m"])

        hp = self.hyper_param

        # ── 1. force close at 15:45 ───────────────────────────────────
        if tod >= hp["force_close_tod"] and pos > 0:
            self._hold_steps  = 0
            self._entry_price = None
            return np.array([1.0, 0.0]), {"entry_point": False, "exit_point": True}

        # ── 2. manage open position ───────────────────────────────────
        if pos > 0:
            self._hold_steps += 1

            # compute pnl since entry
            pnl = 0.0
            if self._entry_price is not None and self._entry_price > 0:
                pnl = (close - self._entry_price) / self._entry_price

            vwap_reverted = abs(vwap_dev) < hp["vwap_revert_thres"]
            time_stop     = self._hold_steps >= int(hp["max_hold_steps"])
            stop_hit      = pnl <= -hp["stop_loss"]
            tp_hit        = pnl >=  hp["take_profit"]

            if vwap_reverted or time_stop or stop_hit or tp_hit:
                self._hold_steps  = 0
                self._entry_price = None
                return np.array([1.0, 0.0]), {"entry_point": False, "exit_point": True}

            return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

        # ── 3. entry logic (flat position only) ──────────────────────
        # time filter 09:45–15:30
        in_time = (585 <= tod <= 930)
        if not in_time:
            return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

        # in-play filter
        in_play = (
            rel_vol   >= hp["rel_vol_thres"]
            and abs(ret_open)  >= hp["ret_open_thres"]
            and range_pct      >= hp["range_thres"]
        )
        if not in_play:
            return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

        # long entry: price stretched below VWAP, oversold, not in freefall
        long_entry = (
            not np.isnan(vwap_dev) and not np.isnan(z20) and not np.isnan(ret_5m)
            and vwap_dev <= -hp["vwap_dev_thres"]
            and z20      <= -hp["z20_thres"]
            and ret_5m   >= -hp["ret_5m_thres"]
        )

        if long_entry:
            self._entry_price = close
            return np.array([1.0, 1.0]), {"entry_point": True, "exit_point": False}

        return np.array([0.0, 0.0]), {"entry_point": False, "exit_point": False}

    @property
    def obs_keys(self):
        return [
            "close",
            "dash@pos",
            self._ta + "tod",
            self._ta + "vwap_dev",
            self._ta + "z20",
            self._ta + "rel_vol",
            self._ta + "ret_since_open",
            self._ta + "range_pct",
            self._ta + "ret_5m",
        ]


# ──────────────────────────────────────────────
#  Helpers (self-contained, no shared dep)
# ──────────────────────────────────────────────

def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 0 or len(a) < w:
        return out
    cs = np.nancumsum(np.where(np.isnan(a), 0.0, a))
    cnt = np.cumsum(~np.isnan(a)).astype(float)
    for i in range(w - 1, len(a)):
        s = cs[i] - (cs[i - w] if i >= w else 0.0)
        k = cnt[i] - (cnt[i - w] if i >= w else 0.0)
        out[i] = s / k if k > 0 else np.nan
    return out


def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 1 or len(a) < w:
        return out
    for i in range(w - 1, len(a)):
        x = a[i - w + 1: i + 1]
        x = x[~np.isnan(x)]
        out[i] = np.std(x, ddof=1) if len(x) > 1 else np.nan
    return out


# ──────────────────────────────────────────────
#  Feature function
# ──────────────────────────────────────────────

@register_function
def features(
    df: pd.DataFrame,
    z_window:   int = 20,
    vol_window: int = 20,
    ret5_window: int = 5,
    prefix: str = "ta@",
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Intraday VWAP mean-reversion features (designed for 1-minute bars).

    Returns
    -------
    out : DataFrame with columns:
        ta@tod             – time of day in minutes since midnight
        ta@vwap            – intraday cumulative VWAP (resets each day)
        ta@vwap_dev        – (close - vwap) / vwap
        ta@z20             – (close - rolling_mean_20) / rolling_std_20
        ta@rel_vol         – volume / rolling_mean_vol_20
        ta@ret_since_open  – (close - session_open) / session_open
        ta@range_pct       – (cum_high - cum_low) / session_open  (so far today)
        ta@ret_5m          – (close - close[t-5]) / close[t-5]
    col_range : dict of [min, max] for observation normalization
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    o = df["open"].values.astype(float)
    n = len(df)

    # ── time of day (minutes since midnight) ──────────────────────────
    ts  = pd.DatetimeIndex(df.index)
    tod = (ts.hour * 60 + ts.minute).astype(float)

    # ── per-day intraday features ─────────────────────────────────────
    dates = ts.normalize()           # day boundary (tz-aware date floors)
    unique_days = dates.unique()

    vwap_arr     = np.full(n, np.nan, dtype=float)
    ret_open_arr = np.full(n, np.nan, dtype=float)
    range_arr    = np.full(n, np.nan, dtype=float)

    for day in unique_days:
        mask = np.asarray(dates == day)
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue

        cv   = c[idx] * v[idx]
        cum_cv = np.cumsum(cv)
        cum_v  = np.cumsum(v[idx])
        vwap_day = np.where(cum_v > 0, cum_cv / cum_v, c[idx])
        vwap_arr[idx] = vwap_day

        open_px = o[idx[0]]
        if open_px > 0:
            ret_open_arr[idx] = (c[idx] - open_px) / open_px
            cum_hi = np.maximum.accumulate(h[idx])
            cum_lo = np.minimum.accumulate(l[idx])
            range_arr[idx] = (cum_hi - cum_lo) / open_px

    # ── cross-bar features ────────────────────────────────────────────
    # VWAP deviation
    vwap_dev = np.where(vwap_arr > 0, (c - vwap_arr) / vwap_arr, np.nan)

    # z-score vs rolling 20-bar mean/std
    mean_z = _rolling_mean(c, z_window)
    std_z  = _rolling_std(c,  z_window)
    z20    = np.where(std_z > 0, (c - mean_z) / std_z, np.nan)

    # relative volume (current bar vs rolling 20-bar avg)
    vol_ma  = _rolling_mean(v, vol_window)
    rel_vol = np.where(vol_ma > 0, v / vol_ma, np.nan)

    # 5-minute return
    ret_5m = np.full(n, np.nan, dtype=float)
    w = int(ret5_window)
    for i in range(w, n):
        if c[i - w] > 0:
            ret_5m[i] = (c[i] - c[i - w]) / c[i - w]

    # ── assemble output ───────────────────────────────────────────────
    out = pd.DataFrame(
        {
            prefix + "tod":            tod,
            prefix + "vwap":           vwap_arr,
            prefix + "vwap_dev":       vwap_dev,
            prefix + "z20":            z20,
            prefix + "rel_vol":        rel_vol,
            prefix + "ret_since_open": ret_open_arr,
            prefix + "range_pct":      range_arr,
            prefix + "ret_5m":         ret_5m,
        },
        index=df.index,
    )

    col_range = {
        prefix + "tod":            [0.0,      1440.0],
        prefix + "vwap":           [0.0,      np.inf],
        prefix + "vwap_dev":       [-np.inf,  np.inf],
        prefix + "z20":            [-np.inf,  np.inf],
        prefix + "rel_vol":        [0.0,      np.inf],
        prefix + "ret_since_open": [-np.inf,  np.inf],
        prefix + "range_pct":      [0.0,      np.inf],
        prefix + "ret_5m":         [-np.inf,  np.inf],
    }

    return out, col_range
