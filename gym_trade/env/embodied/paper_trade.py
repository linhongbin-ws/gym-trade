import gym
import pandas as pd
import numpy as np
from typing import List
from gym_trade.env.embodied.base import BaseEnv
from gym.utils import seeding


class PaperTrade(BaseEnv):
    def __init__(self, 
                    df: pd.DataFrame , # list of csv file names or list of df 
                    interval: str,
                    obs_keys: List[str] = ["position_ratio","open","close","high","low"],
                    init_balance: float =  1e6,
                    commission_type: str   = "free", 
                    reward_type: str = 'sparse',
                    dash_keys: List[str] =  ['pos','cash', 'balance','pnl'],
                    action_deadzone: float = 0.01,
                    action_on: str = "close_t_minus_1", # apply action on which time frame: [close_t_minus_1, open_t]
                    col_range_dict: dict = None,
                    ):
        
        super().__init__()
        self._seed = None
        self._t = 0

        assert interval in ["1d", "1m"], f"interval {interval} not supported in paper trade"
        assert action_on in ["close_t_minus_1", "open_t"], action_on
        self._df = df.copy()
        self._interval = interval
        self._obs_keys = obs_keys
        self._init_balance = init_balance
        self._commission_type = commission_type
        self._reward_type = reward_type
        self._dash_keys = dash_keys
        self._action_deadzone = action_deadzone 
        self._action_on = action_on
        self._col_range_dict = col_range_dict

        default_col_range_dict = {'close': [0, np.inf], 
                                'high': [0, np.inf],  
                                'low': [0, np.inf],  
                                'open': [0, np.inf],  
                                'volume': [0, np.inf], 
                                'dash@pos': [0, np.inf],
                                'dash@cash': [-np.inf, np.inf],
                                'dash@balance': [-np.inf, np.inf],
                                'dash@pnl': [-np.inf, np.inf],}
        for k in default_col_range_dict:
            if k not in self._col_range_dict:
                self._col_range_dict[k] = default_col_range_dict[k]


        self.seed(0)
        self._init_or_reset_cb() # init variables

    #====== gym api ========
    def reset(self):
        self._init_or_reset_cb()
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        # action clip to action_space range
        action = np.clip(action,-1,1)

        self._t += 1
        assert self._t <= (len(self._df.index) - 1), f"self._t: {self._t} is out of df index: {len(self._df.index) - 1} "

        self._step_action(action) 

        reward = self._get_reward()
        info = self._get_info()
        obs = self._get_obs()
        done = self._t >= (len(self._df.index)-1) 
        return obs, reward, done, info
    
    def seed(self, seed):
        self._seed = seed
        self._df_rng, seed = seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):  
        # action[0]: range [-1,0] hold same position, range (0,1] change position according to action[1]
        # action[1]: range [-1, 1] mapping from minimum position to maximum position
        return gym.spaces.Box(low=-1, high=1, shape=(2,))
    
    @property
    def observation_space(self):
        obs = {}
        for v in self._obs_keys:
            if v in self._col_range_dict:
                obs[v] = gym.spaces.Box(low=self._col_range_dict[v][0], high=self._col_range_dict[v][1], shape=(1,), dtype=float)
            else:
                raise NotImplementedError(f"col_range_dict for {v} is not found")
        return gym.spaces.Dict(obs)

    
    #====== other ==============
    def _step_action(self, action):
        
        close = self._df['close'].iloc[self._t]
        open = self._df['open'].iloc[self._t]
        cash_prv = self._df['dash@cash'].iloc[self._t-1]
        pos_prv = self._df['dash@pos'].iloc[self._t-1]
        if self._action_on == "close_t_minus_1":
            action_price = self._df['close'].iloc[self._t-1]
            action_balance = self._df['dash@balance'].iloc[self._t-1]
        elif self._action_on == "open_t":
            action_price = open 
            action_balance = cash_prv + pos_prv * open

        # action[0] decide whether hold, action[1] control the position 
        # change position according to action[1]
        if action[0] >0 :
            k, b = self._get_commision_coff()
            max_pos = np.floor( (cash_prv - b) / (k + action_price) + pos_prv )
            min_pos = np.ceil( (cash_prv - 2 * action_balance) / (k+ action_price) + pos_prv ) 
            if action[1] > 0:
                target_pos = np.floor(max_pos * action[1])
                
            elif action[1] < 0:
                target_pos = np.ceil(min_pos * action[1])
            else:
                target_pos = 0

            stock_pos_change = target_pos - pos_prv
            commision = k * np.abs(stock_pos_change) + b 
            self._df['dash@pos'].iloc[self._t] = target_pos
            self._df['dash@cash'].iloc[self._t] = cash_prv - stock_pos_change * action_price - commision


            # if stock_pos > 0: 
            #     self._df['dash@cash'].iloc[self._t] = cash_prv - action_cash
            #     self._df['dash@pos'].iloc[self._t] = pos_prv + stock_pos 
            # else:
            #     self._df['dash@cash'].iloc[self._t] = cash_prv 
                # self._df['dash@pos'].iloc[self._t] = pos_prv  



        # hold
        else:
            self._df['dash@cash'].iloc[self._t] = cash_prv
            self._df['dash@pos'].iloc[self._t] = pos_prv
            stock_pos_change = 0


        self._df['dash@balance'].iloc[self._t] = self._df['dash@pos'].iloc[self._t] * close  \
                                                        + self._df['dash@cash'].iloc[self._t]
        self._df['dash@pnl'].iloc[self._t] = (self._df['dash@balance'].iloc[self._t] - self._init_balance)\
                                                       / self._init_balance 
        # self._df['action'].iloc[self._t-1] = action 

        info = {}
        info['stock_pos_change'] = stock_pos_change


        assert self._df['dash@cash'].iloc[self._t] >=0, f"Cash cannot be negative: {self._df['dash@cash'].iloc[self._t]}"
        assert self._df['dash@pos'].iloc[self._t] >=0, f"pos cannot be negative: {self._df['dash@pos'].iloc[self._t]}, do not suppport shorting yet"


    def _get_reward(self):
        sparse_update_condition = self._df['dash@pos'].iloc[self._t] ==0 and self._df['dash@pos'].iloc[self._t-1]>0
        if sparse_update_condition and self._reward_type == "sparse":
            return 0.0
        
        return self._df['dash@pnl'].iloc[self._t] - self._df['dash@pnl'].iloc[self._t - 1]
    
    def _init_or_reset_cb(self):
        """
        call each reset() or init()
        """
        self._t = 0
        self._df["action"] = np.nan

        # create dash columns
        for k in self._dash_keys:
            self._df["dash@" + k] = np.nan # fill with nan

        bal = float(self._init_balance)
        self._df['dash@balance'].iloc[0] = bal # all balance at time 0
        self._df['dash@cash'].iloc[0] =bal # all cash at time 0
        self._df['dash@pos'].iloc[0] = 0.0 # no stock position at time 0
        self._df['dash@pnl'].iloc[0] = 0.0

        # check if obs key exist
        for k in self._obs_keys:
            assert k in self._df.columns, f"obs key {k} is not in dataframe, columns: {self._df.columns}"

    def _get_info(self):
        return {}


    def _get_obs(self):
        obs = {}
        for k in self._obs_keys:
            obs[k] = self._df.iloc[self._t][k]
        return obs

    
    def _get_commision_coff(self):
        """
        we assume the commision is linear model: commsion = k* buy_pos + b
        """
        if self._commission_type == "futu":
            min_commision = 0.99
            commision = 0.0049
            platform = 0.005
            min_platform = 1

            # I make simple assumption, ignore minimum thres

            k = commision + platform
            b = 0
            return k, b
            # return np.clip(pos*commision, min_commision, None) + np.clip(pos*platform,min_platform, None) 
        elif self._commission_type == "free":
            k = 0
            b = 0
            return k, b 
        else:
            raise NotImplementedError

    @property
    def df(self):
        return self._df
    

    
    @property
    def pnl(self):
        pnl = self.df.iloc[self._t ]["dash@pnl"]
        return pnl
