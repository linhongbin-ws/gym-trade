import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from gym_trade.policy.registry import register_policy, register_function
from gym_trade.policy.base import BasePolicy

@register_policy
class Policy(BasePolicy):
    def __init__(self,  **kwargs):
        hyper_param_range = {
            # --- 入场条件 ---
            # breakout_strength = (close - donchian_high_prev)/ATR
            "breakout_strength_thres": (0.20, -0.50, 3.00),  # >=0 表示已突破；>0.2 表示突破更“干净”
            "vol_z_thres": (0.30, -0.50, 5.00),              # vol_z = vol/ma - 1；0.3=较明显放量
            "close_pos_thres": (0.65, 0.00, 1.00),           # 当天收盘靠近高点，减少假突破

            # 可选：只做“压缩后突破”（squeeze_pct 越大，代表当前宽度高于历史极值；这里我们用“近期很紧”
            # 做法：要求当前 bb_width 接近 squeeze_window 内最小值 => squeeze_pct 接近 1
            "use_squeeze_filter": (1, 0, 1),
            "squeeze_pct_max": (1.30, 1.00, 5.00),           # squeeze_pct <= 1.3 视为“较紧”

            # --- 退出/风控 ---
            # 1) ATR 跟踪止损：用 close - k*ATR 的 trailing stop
            "atr_trail_k": (3.0, 0.5, 8.0),

            # 2) 趋势破坏：跌回 Donchian 下轨附近（pullback_depth <= 0 表示跌破下轨）
            "breakdown_pullback_thres": (-0.20, -3.00, 1.00),

            # --- 稳健性控制 ---
            "min_hold_steps": (5, 0, 100),            # 突破策略一般不需要很长最小持仓期
            "cooldown_steps": (3, 0, 50),
            "exit_confirm_steps": (2, 1, 10),         # 连续N次触发才卖，减少单日长下影误杀
        }
        super().__init__(hyper_param_range=hyper_param_range)

        self._ta_prefix  = "ta@"

        # state
        self._hold_steps = 0
        self._cooldown = 0
        self._exit_count = 0

        # trailing stop state
        self._trail_stop = None  # float

    def __call__(self, obs, **kwargs):
        
        pos = obs["dash@pos"]

        breakout_strength = float(obs[self._ta_prefix  + "breakout_strength"])
        pullback_depth = float(obs[self._ta_prefix  + "pullback_depth"])
        atr = float(obs[self._ta_prefix  + "atr"])
        vol_z = float(obs[self._ta_prefix  + "vol_z"])
        close_pos = float(obs[self._ta_prefix  + "close_pos_in_range"])
        squeeze_pct = float(obs[self._ta_prefix  + "squeeze_pct"])
        close = float(obs.get("dash@close", np.nan))  # 如果你的框架没有 dash@close，就删掉并用别的 close 字段

        # counters
        if pos > 0:
            self._hold_steps += 1
        else:
            self._hold_steps = 0
            self._trail_stop = None
            self._exit_count = 0

        if self._cooldown > 0:
            self._cooldown -= 1

        # --- filters ---
        can_enter = (self._cooldown == 0)

        squeeze_ok = True
        if int(self.hyper_param["use_squeeze_filter"]) == 1:
            # “紧”=> squeeze_pct 接近 1（当前宽度接近历史最小）
            squeeze_ok = (squeeze_pct <= self.hyper_param["squeeze_pct_max"])

        # --- entry ---
        entry = (
            can_enter
            and squeeze_ok
            and (breakout_strength >= self.hyper_param["breakout_strength_thres"])
            and (vol_z >= self.hyper_param["vol_z_thres"])
            and (close_pos >= self.hyper_param["close_pos_thres"])
        )

        # --- exit ---
        # 1) trailing stop update: stop = max(stop, close - k*ATR)
        # 注意：需要 close 值。若你没有 close 的 obs key，就把 trailing stop 改成基于 donchian_low_prev 的止损。
        if pos > 0 and not np.isnan(close) and not np.isnan(atr):
            new_stop = close - self.hyper_param["atr_trail_k"] * atr
            if self._trail_stop is None or np.isnan(self._trail_stop):
                self._trail_stop = new_stop
            else:
                self._trail_stop = max(self._trail_stop, new_stop)

        stop_hit = False
        if pos > 0 and self._trail_stop is not None and not np.isnan(close):
            stop_hit = (close <= self._trail_stop)

        # 2) breakdown: 价格显著回撤到下轨（或接近跌破）
        breakdown = (pullback_depth <= self.hyper_param["breakdown_pullback_thres"])

        exit_raw = stop_hit or breakdown

        # --- decision ---
        if pos == 0:
            if entry:
                return np.array([1, 1]), {"entry_point": True, "exit_point": False}
            return np.array([0, 0]), {"entry_point": False, "exit_point": False}

        # pos > 0
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
            self._trail_stop = None
            return np.array([1, 0]), {"entry_point": False, "exit_point": True}

        return np.array([0, 0]), {"entry_point": False, "exit_point": False}

    @property
    def obs_keys(self):
        keys = [
            self._ta_prefix  + "donchian_high_prev",
            self._ta_prefix  + "donchian_low_prev",
            self._ta_prefix  + "breakout_strength",
            self._ta_prefix  + "pullback_depth",
            self._ta_prefix  + "atr",
            self._ta_prefix  + "atr_pct",
            self._ta_prefix  + "vol_z",
            self._ta_prefix  + "close_pos_in_range",
            self._ta_prefix  + "bb_width",
            self._ta_prefix  + "squeeze_pct",
        ]
        # 如果你框架里能提供 close，建议加上用于 trailing stop
        keys += ["dash@pos"] 
        return keys



def _rolling_max(a: np.ndarray, w: int) -> np.ndarray:
    # simple, fast enough baseline; replace with bottleneck if you have it
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 0 or len(a) < w:
        return out
    for i in range(w - 1, len(a)):
        out[i] = np.nanmax(a[i - w + 1:i + 1])
    return out

def _rolling_min(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 0 or len(a) < w:
        return out
    for i in range(w - 1, len(a)):
        out[i] = np.nanmin(a[i - w + 1:i + 1])
    return out

def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 0 or len(a) < w:
        return out
    c = np.nancumsum(np.where(np.isnan(a), 0.0, a))
    n = np.cumsum(~np.isnan(a)).astype(float)
    for i in range(w - 1, len(a)):
        s = c[i] - (c[i - w] if i - w >= 0 else 0.0)
        k = n[i] - (n[i - w] if i - w >= 0 else 0.0)
        out[i] = s / k if k > 0 else np.nan
    return out

def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if w <= 1 or len(a) < w:
        return out
    for i in range(w - 1, len(a)):
        x = a[i - w + 1:i + 1]
        x = x[~np.isnan(x)]
        out[i] = np.std(x, ddof=1) if len(x) > 1 else np.nan
    return out

def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.full(len(close), np.nan, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
    return tr

def _ema(a: np.ndarray, span: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if span <= 1:
        return a.astype(float)
    alpha = 2.0 / (span + 1.0)
    s = np.nan
    for i, v in enumerate(a):
        if np.isnan(v):
            out[i] = s
            continue
        if np.isnan(s):
            s = v
        else:
            s = alpha * v + (1 - alpha) * s
        out[i] = s
    return out

@register_function
def features( # fts=features
    df: pd.DataFrame,
    donchian_window: int = 20,
    atr_window: int = 14,
    vol_window: int = 20,
    bb_window: int = 20,
    bb_k: float = 2.0,
    squeeze_window: int = 60,
    prefix: str = "ta@",
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Features for trend breakout on daily OHLCV:
    - donchian_high_prev: previous N-day highest high (excluding today)
    - donchian_low_prev:  previous N-day lowest low (excluding today)
    - breakout_strength:  (close - donchian_high_prev) / atr  (>=0 => above breakout line)
    - pullback_depth:     (close - donchian_low_prev) / atr   (<=0 => below breakdown line)
    - atr_pct:            ATR / close
    - vol_z:              (volume / vol_ma) - 1
    - close_pos_in_range: (close - low) / (high - low)  (0~1)
    - bb_width:           (upper-lower)/mid
    - squeeze_pct:        bb_width / rolling_min(bb_width, squeeze_window)  (>=1)
    """
    out_range = {
        prefix + "donchian_high_prev": [-np.inf, np.inf],
        prefix + "donchian_low_prev":  [-np.inf, np.inf],
        prefix + "breakout_strength":  [-np.inf, np.inf],
        prefix + "pullback_depth":     [-np.inf, np.inf],
        prefix + "atr":                [0, np.inf],
        prefix + "atr_pct":            [0, np.inf],
        prefix + "vol_z":              [-np.inf, np.inf],
        prefix + "close_pos_in_range": [0, 1],
        prefix + "bb_width":           [0, np.inf],
        prefix + "squeeze_pct":        [1, np.inf],
    }
    
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)

    # Donchian prev (exclude today): compute rolling max/min then shift by 1
    don_hi = _rolling_max(h, int(donchian_window))
    don_lo = _rolling_min(l, int(donchian_window))
    don_hi_prev = np.roll(don_hi, 1); don_hi_prev[0] = np.nan
    don_lo_prev = np.roll(don_lo, 1); don_lo_prev[0] = np.nan

    # ATR (EMA of TR)
    tr = _true_range(h, l, c)
    atr = _ema(tr, int(atr_window))
    atr_pct = atr / c

    # breakout strength normalized by ATR
    breakout_strength = (c - don_hi_prev) / atr
    pullback_depth = (c - don_lo_prev) / atr

    # volume surge
    vol_ma = _rolling_mean(v, int(vol_window))
    vol_z = (v / vol_ma) - 1.0

    # close position in daily range
    rng = (h - l)
    close_pos = np.where(rng > 0, (c - l) / rng, np.nan)
    close_pos = np.clip(close_pos, 0.0, 1.0)

    # Bollinger width as squeeze proxy
    mid = _rolling_mean(c, int(bb_window))
    sd = _rolling_std(c, int(bb_window))
    upper = mid + bb_k * sd
    lower = mid - bb_k * sd
    bb_width = np.where(mid > 0, (upper - lower) / mid, np.nan)

    # squeeze ratio relative to rolling min width
    bb_min = _rolling_min(bb_width, int(squeeze_window))
    squeeze_pct = bb_width / bb_min

    out = pd.DataFrame({
        prefix + "donchian_high_prev": don_hi_prev,
        prefix + "donchian_low_prev":  don_lo_prev,
        prefix + "breakout_strength":  breakout_strength,
        prefix + "pullback_depth":     pullback_depth,
        prefix + "atr":                atr,
        prefix + "atr_pct":            atr_pct,
        prefix + "vol_z":              vol_z,
        prefix + "close_pos_in_range": close_pos,
        prefix + "bb_width":           bb_width,
        prefix + "squeeze_pct":        squeeze_pct,
    }, index=df.index)

    return out, out_range
