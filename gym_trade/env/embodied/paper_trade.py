import gym
import pandas as pd
import numpy as np
from os.path import isfile
from typing import List, Union
import pathlib
from gym_trade.env.embodied.base import BaseEnv
from gym.utils import seeding


class PaperTrade(BaseEnv):
    def __init__(self, 
                    df_list: List[pd.DataFrame] , # list of csv file names or list of df 
                    interval: str,
                    obs_keys: List[str] = ["position_ratio","open","close","high","low"],
                    init_balance: float =  1e6,
                    commission_type: str   = "free", 
                    reward_type: str = 'sparse',
                    dash_keys: List[str] =  ['pos','cash', 'balance','pnl'],
                    action_deadzone: float = 0.01,
                    action_on: str = "close_t_minus_1", # apply action on which time frame: [close_t_minus_1, open_t]
                    ):
        
        super().__init__()
        self._seed = None
        self._t = 0

        assert interval in ["1d", "1m"], f"interval {interval} not supported in paper trade"
        assert action_on in ["close_t_minus_1", "open_t"], action_on
        self._df_list = df_list
        self._interval = interval
        self._obs_keys = obs_keys
        self._init_balance = init_balance
        self._commission_type = commission_type
        self._reward_type = reward_type
        self._dash_keys = dash_keys
        self._action_deadzone = action_deadzone 
        self._action_on = action_on


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
        #  dz~1: buy, -1~-dz: sell, -dz~dz: hold=0 ; dz is action deadzone 
        return gym.spaces.Box(low=-1, high=1, shape=(1,))
    
    @property
    def observation_space(self):
        obs = {}
        for v in self._obs_keys:
            if v=="position_ratio":
                obs[v] = gym.spaces.Box(low=0,high=1,shape=(1,),dtype=float)
            elif v=="timestep":
                obs[v] = gym.spaces.Box(low=0,high=389,shape=(1,),dtype=int)
            elif v in ["open","close","high","low"]:
                obs[v] = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(1,),dtype=float)
            else:
                raise NotImplementedError
        return gym.spaces.Dict(obs)

    
    #====== other ==============
    def _step_action(self, action):
        
        close = self._df['close'].iloc[self._t]
        open = self._df['open'].iloc[self._t]
        cash_prv = self._df['dash@cash'].iloc[self._t-1]
        pos_prv = self._df['dash@pos'].iloc[self._t-1]
        if self._action_on == "close_t_minus_1":
            action_price = self._df['close'].iloc[self._t-1]
        elif self._action_on == "open_t":
            action_price = open 

        
        # buy
        if action >= self._action_deadzone and cash_prv != 0:
            action_cash = cash_prv * action
            k, b = self._get_commision_coff()
            stock_pos = np.floor((action_cash - b) / (k + action_price) ) # assume commision is linear model

            if stock_pos > 0: 
                self._df['dash@cash'].iloc[self._t] = cash_prv - action_cash
                self._df['dash@pos'].iloc[self._t] = pos_prv + stock_pos 
            else:
                self._df['dash@cash'].iloc[self._t] = cash_prv 
                self._df['dash@pos'].iloc[self._t] = pos_prv  

        
        # sell
        elif action <= -self._action_deadzone and pos_prv!=0:
            k, b = self._get_commision_coff()
            stock_pos = np.floor(pos_prv * np.abs(action))
            commision_cash = stock_pos * k + b
            sell_cash = stock_pos * action_price - commision_cash
            if sell_cash >=0: # if sell price after commision is negative, we keep it not sell
                self._df['dash@cash'].iloc[self._t] = cash_prv + sell_cash
                self._df['dash@pos'].iloc[self._t] = pos_prv - stock_pos
            else:
                self._df['dash@cash'].iloc[self._t] = cash_prv 
                self._df['dash@pos'].iloc[self._t] = pos_prv  

        # hold
        else:
            self._df['dash@cash'].iloc[self._t] = cash_prv
            self._df['dash@pos'].iloc[self._t] = pos_prv
            stock_pos = 0


        self._df['dash@balance'].iloc[self._t] = self._df['dash@pos'].iloc[self._t] * close  \
                                                        + self._df['dash@cash'].iloc[self._t]
        self._df['dash@pnl'].iloc[self._t] = (self._df['dash@balance'].iloc[self._t] - self._init_balance)\
                                                       / self._init_balance 
        self._df['action'].iloc[self._t-1] = action 

        info = {}
        info['stock_pos_change'] = stock_pos


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

        # randomize df index for each init
        self._df_idx = 0 if len(self._df_list)==1 else self._rng_csv_idx.randint(0, len(self._df_list)-1) 
        self._df = self._df_list[ self._df_idx].copy()


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
            assert k in self._df.columns, f"obs key {k} is not in dataframe"

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
