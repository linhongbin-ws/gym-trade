import gym
import pandas as pd
import numpy as np
from typing import List
from gym_trade.env.embodied.base import BaseEnv
from gym.utils import seeding


class PaperTrade(BaseEnv):
    def __init__(self,
                    df: pd.DataFrame,
                    interval: str,
                    obs_keys: List[str] = ["position_ratio","open","close","high","low"],
                    init_balance: float = 1e6,
                    commission_type: str = "free",
                    reward_type: str = 'sparse',
                    dash_keys: List[str] = ['pos','cash', 'balance','pnl'],
                    action_deadzone: float = 0.01,
                    action_on: str = "close_t_minus_1",
                    eod_close: bool = False,
                    col_range_dict: dict = None,
                    ):
        super().__init__()
        self._seed = None
        self._t = 0

        assert interval in ["1d", "1m"], f"interval {interval} not supported"
        assert action_on in ["close_t_minus_1", "open_t"], action_on

        self._df = df  # kept for vis / pnl property only
        self._interval = interval
        self._obs_keys = obs_keys
        self._init_balance = float(init_balance)
        self._commission_type = commission_type
        self._reward_type = reward_type
        self._dash_keys = dash_keys
        self._action_deadzone = action_deadzone
        self._action_on = action_on
        self._eod_close = eod_close and (interval == "1m")
        self._col_range_dict = col_range_dict

        default_col_range_dict = {
            'close':        [0, np.inf],
            'high':         [0, np.inf],
            'low':          [0, np.inf],
            'open':         [0, np.inf],
            'volume':       [0, np.inf],
            'dash@pos':     [0, np.inf],
            'dash@cash':    [-np.inf, np.inf],
            'dash@balance': [-np.inf, np.inf],
            'dash@pnl':     [-np.inf, np.inf],
        }
        for k in default_col_range_dict:
            if k not in self._col_range_dict:
                self._col_range_dict[k] = default_col_range_dict[k]

        # Pre-convert market data columns to numpy (read-only, never changes)
        self._np = {col: df[col].to_numpy(dtype=np.float64) for col in df.columns}
        self._n = len(df)

        # Pre-build obs key list (dash@ keys read from live numpy arrays below)
        for k in obs_keys:
            assert k in self._np or k.startswith('dash@'), \
                f"obs key {k} not in dataframe columns: {list(df.columns)}"

        # Precompute EOD flags: True on the last bar of each trading session
        # Used to force-close positions before overnight gap (1m only)
        if self._eod_close:
            dates = np.array([pd.Timestamp(ts).date() for ts in df.index])
            self._is_eod = np.zeros(self._n, dtype=bool)
            self._is_eod[:-1] = dates[:-1] != dates[1:]
            self._is_eod[-1] = True  # last bar is always EOD
        else:
            self._is_eod = np.zeros(self._n, dtype=bool)

        self.seed(0)
        self._init_dash_arrays()

    # ====== gym api ========
    def reset(self):
        self._init_dash_arrays()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)
        self._t += 1
        assert self._t <= self._n - 1

        # Force close at end of session (bar t-1 was the last bar of the day)
        if self._eod_close and self._is_eod[self._t - 1] and self._dash_pos[self._t - 1] > 0:
            action = np.array([1.0, 0.0])

        self._step_action(action)

        reward = self._get_reward()
        obs = self._get_obs()
        done = self._t >= self._n - 1
        return obs, reward, done, {}

    def seed(self, seed):
        self._seed = seed
        self._df_rng, seed = seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def observation_space(self):
        obs = {}
        for v in self._obs_keys:
            if v in self._col_range_dict:
                obs[v] = gym.spaces.Box(
                    low=self._col_range_dict[v][0],
                    high=self._col_range_dict[v][1],
                    shape=(1,), dtype=float)
            else:
                raise NotImplementedError(f"col_range_dict for {v} not found")
        return gym.spaces.Dict(obs)

    # ====== internals ======
    def _init_dash_arrays(self):
        self._t = 0
        n = self._n
        self._dash_pos     = np.zeros(n, dtype=np.float64)
        self._dash_cash    = np.zeros(n, dtype=np.float64)
        self._dash_balance = np.zeros(n, dtype=np.float64)
        self._dash_pnl     = np.zeros(n, dtype=np.float64)

        self._dash_balance[0] = self._init_balance
        self._dash_cash[0]    = self._init_balance
        self._dash_pos[0]     = 0.0
        self._dash_pnl[0]     = 0.0

    def _step_action(self, action):
        t = self._t
        close = self._np['close'][t]
        open_ = self._np['open'][t]
        cash_prv = self._dash_cash[t - 1]
        pos_prv  = self._dash_pos[t - 1]

        if self._action_on == "close_t_minus_1":
            action_price   = self._np['close'][t - 1]
            action_balance = self._dash_balance[t - 1]
        else:  # open_t
            action_price   = open_
            action_balance = cash_prv + pos_prv * open_

        if action[0] > 0:
            k, b = self._get_commission_coeff()
            max_pos = np.floor((cash_prv - b) / (k + action_price) + pos_prv)
            min_pos = np.ceil((cash_prv - 2 * action_balance) / (k + action_price) + pos_prv)

            if action[1] > 0:
                target_pos = np.floor(max_pos * action[1])
            elif action[1] < 0:
                target_pos = np.ceil(min_pos * action[1])
            else:
                target_pos = 0.0

            delta = target_pos - pos_prv
            commission = k * abs(delta) + b
            self._dash_pos[t]  = target_pos
            self._dash_cash[t] = cash_prv - delta * action_price - commission
        else:
            self._dash_pos[t]  = pos_prv
            self._dash_cash[t] = cash_prv

        self._dash_balance[t] = self._dash_pos[t] * close + self._dash_cash[t]
        self._dash_pnl[t]     = (self._dash_balance[t] - self._init_balance) / self._init_balance

        assert self._dash_cash[t] >= 0, f"Cash negative: {self._dash_cash[t]}"
        assert self._dash_pos[t]  >= 0, f"Pos negative: {self._dash_pos[t]}"

    def _get_reward(self):
        t = self._t
        if self._reward_type == "sparse":
            # only reward at exit (pos just went to 0)
            if self._dash_pos[t] == 0 and self._dash_pos[t - 1] > 0:
                return self._dash_pnl[t] - self._dash_pnl[t - 1]
            return 0.0
        return self._dash_pnl[t] - self._dash_pnl[t - 1]

    def _get_obs(self):
        t = self._t
        obs = {}
        for k in self._obs_keys:
            if k.startswith('dash@'):
                dash_name = k[5:]  # strip 'dash@'
                obs[k] = getattr(self, f'_dash_{dash_name}')[t]
            else:
                obs[k] = self._np[k][t]
        return obs

    def _get_commission_coeff(self):
        if self._commission_type == "futu":
            k = 0.0049 + 0.005
            b = 0
            return k, b
        elif self._commission_type == "free":
            return 0.0, 0.0
        else:
            raise NotImplementedError

    @property
    def df(self):
        # Sync dash arrays back to df for vis/downstream use
        self._df['dash@pos']     = self._dash_pos
        self._df['dash@cash']    = self._dash_cash
        self._df['dash@balance'] = self._dash_balance
        self._df['dash@pnl']     = self._dash_pnl
        return self._df

    @property
    def pnl(self):
        return self._dash_pnl[self._t]

    @property
    def is_eod(self) -> np.ndarray:
        """Boolean array marking the last bar of each trading session."""
        return self._is_eod
