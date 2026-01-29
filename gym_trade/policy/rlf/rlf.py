from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as np

@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str], **kwargs):
        hyper_param_range = {
            "buy_thres": (0.12, 0.01, 0.40),
            "sell_thres": (0.06, 0.00, 0.30),   # 注意：应 < buy_thres（滞回）
            "r2_thres": (0.35, 0.05, 0.95),
            "t_thres": (2.0, 0.5, 5.0),
            "min_hold_steps": (5, 0, 40),
            "cooldown_steps": (3, 0, 40),
        }
        super().__init__(hyper_param_range=hyper_param_range)

        # 内部状态：持仓后计数 / 冷却计数
        self._hold_steps = 0
        self._cooldown = 0

    def __call__(self, obs, **kwargs):
        prefix = "rlf_ma240_60@"
        pos = obs["dash@pos"]

        slope_pct = obs[prefix + "slope_pct_annual"]
        slope = obs[prefix + "slope"]
        r2 = obs[prefix + "r2"]
        t = obs[prefix + "t"]

        # 更新计数器
        if pos > 0:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        if self._cooldown > 0:
            self._cooldown -= 1

        # 趋势显著性过滤（减少噪声交易）
        trend_quality = (r2 >= self.hyper_param["r2_thres"]) and (abs(t) >= self.hyper_param["t_thres"])

        # 更严格的入场：趋势强 + 方向确认
        buy_signal = trend_quality and (slope_pct >= self.hyper_param["buy_thres"]) and (slope > 0)

        # 出场：趋势走弱（滞回）或方向翻转
        sell_signal = (slope_pct <= self.hyper_param["sell_thres"]) or (slope < 0)

        # 决策
        if pos == 0:
            if self._cooldown == 0 and buy_signal:
                return 1
            return 0

        # pos > 0
        if self._hold_steps < self.hyper_param["min_hold_steps"]:
            return 0
        if sell_signal:
            self._cooldown = int(self.hyper_param["cooldown_steps"])
            return -1
        return 0

    @property
    def obs_keys(self):
        prefix = "rlf_ma240_60@"
        keys = [prefix + k for k in ["slope", "intercept", "r2", "t", "slope_pct_annual"]]
        keys += ["dash@pos"]
        return keys
