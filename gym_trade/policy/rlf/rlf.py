from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy
import numpy as np

@register_policy
class Policy(BasePolicy):
    def __init__(self, obs_keys: list[str], **kwargs):
        # (default, low, high) 形式按你现有框架习惯写
        hyper_param_range = {
            # 入场：初次入场更严格；回补更宽松（通常 reentry < buy）
            "buy_thres": (0.14, 0.02, 0.60),
            "reentry_thres": (0.06, -0.05, 0.40),

            # 退出：更严格（通常 <= 0，避免牛市中被洗下车）
            "exit_thres": (-0.02, -0.30, 0.15),

            # 趋势质量过滤：入场/退出分开（退出门槛通常更低，保证能及时 risk-off）
            "r2_buy": (0.20, 0.00, 0.95),
            "t_buy": (1.0, 0.0, 6.0),
            "r2_exit": (0.05, 0.00, 0.95),
            "t_exit": (0.0, 0.0, 6.0),

            # 稳健性控制
            "min_hold_steps": (30, 0, 400),      # 买入后至少持有这么久才允许卖
            "cooldown_steps": (3, 0, 200),       # 卖出后冷却，避免立刻追涨买回
            "exit_confirm_steps": (3, 1, 20),    # 退出信号连续出现 N 次才卖（减少假退出）
        }
        super().__init__(hyper_param_range=hyper_param_range)

        # 策略内部状态（假设 policy 对象在 episode 内常驻）
        self._hold_steps = 0
        self._cooldown = 0
        self._exit_count = 0

    def __call__(self, obs, **kwargs):
        prefix = "rlf_ma240_60@"
        pos = obs["dash@pos"]

        slope_pct = float(obs[prefix + "slope_pct_annual"])
        slope = float(obs[prefix + "slope"])
        r2 = float(obs[prefix + "r2"])
        t = float(obs[prefix + "t"])

        # 更新计数器
        if pos > 0:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        if self._cooldown > 0:
            self._cooldown -= 1

        # --- 信号定义 ---
        # 趋势质量：入场更严格；退出更宽松（避免错过风险退出）
        quality_buy = (r2 >= self.hyper_param["r2_buy"]) and (abs(t) >= self.hyper_param["t_buy"])
        quality_exit = (r2 >= self.hyper_param["r2_exit"]) and (abs(t) >= self.hyper_param["t_exit"])

        # 方向确认（避免纯噪声）
        up_confirm = (slope > 0)
        down_confirm = (slope < 0)

        # 入场：
        # - 空仓时用 buy_thres（更严格）
        # - 冷却后回补用 reentry_thres（更宽松，减少错过 V 形反弹）
        can_enter = (self._cooldown == 0)
        enter_fresh = can_enter and quality_buy and up_confirm and (slope_pct >= self.hyper_param["buy_thres"])
        enter_reentry = can_enter and quality_buy and up_confirm and (slope_pct >= self.hyper_param["reentry_thres"])

        # 退出（risk-off）：
        # 趋势显著走坏 + 方向转下；并用连续确认减少误退出
        exit_raw = quality_exit and down_confirm and (slope_pct <= self.hyper_param["exit_thres"])

        # --- 决策 ---
        if pos == 0:
            # 空仓：优先用“初次入场”，否则用“回补入场”
            self._exit_count = 0
            if enter_fresh or enter_reentry:
                return 1
            return 0

        # pos > 0：持仓
        # 最小持仓期内不卖（减少被洗下车的机会成本）
        if self._hold_steps < int(self.hyper_param["min_hold_steps"]):
            self._exit_count = 0
            return 0

        # 连续确认退出
        if exit_raw:
            self._exit_count += 1
        else:
            # 退出条件消失就清零，避免“断断续续”触发
            self._exit_count = 0

        if self._exit_count >= int(self.hyper_param["exit_confirm_steps"]):
            # 卖出后进入冷却
            self._cooldown = int(self.hyper_param["cooldown_steps"])
            self._exit_count = 0
            return -1

        return 0

    @property
    def obs_keys(self):
        prefix = "rlf_ma240_60@"
        keys = [prefix + k for k in ["slope", "intercept", "r2", "t", "slope_pct_annual"]]
        keys += ["dash@pos"]
        return keys
