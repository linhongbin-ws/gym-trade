import gym
from gym import spaces
import pandas as pd
import numpy as np
from os.path import isfile
import random
from typing import List, Union




#=== gym_datatrade package
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from gym_trade.tool import ta
from gym_trade.tool.common import get_csv_dir


class US_Stock_Env(gym.Env):
    def __init__(self, csv_root_dir, 
                    init_balance=100,
                    commission_rate = 3e-4, 
                    reward_type='pnl',
                    obs_keys=["stat_position_ratio"],
                    stat_keys=['stat_balance', 'stat_profit_ratio', 'stat_pnl'],
                    verbose=0,
                    **kwargs,
                    ):
        self._csv_root_dir = csv_root_dir
        self._init_balance = init_balance
        self._commision_rate = commission_rate
        self._reward_type = reward_type
        self._obs_keys = obs_keys
        self._stat_keys = stat_keys
        self._set_clock_func = lambda date, hour, minute: date.replace(hour=hour, minute=minute)
        self._verbose = verbose
        
        self. _update_csv_dir(self, self._csv_root_dir)

        self._seed = 0
        self._init_var()

    def reset(self):
        self._init_var()
        obs = self._get_obs()
        return obs
    


    def step(self, action):
        self._timestep +=1
        assert self._timestep <=self._df.shape[0] - 1

        self._update_stats(action) 

        #======= outputs========
        info = self._get_info()
        obs = self._get_obs()
        done = len(info['done']) !=0
        reward = {} 

        return obs, reward, done, info

    def _update_csv_dir(self, dir):
        if isinstance(dir, str): # extract csv file path
            self._csv_list = get_csv_dir(dir)
        elif isinstance(csv_dir, list):
            self._csv_list = csv_dir
        else:
            raise NotImplementedError
        assert len(self._csv_list)!=0, f"data_dir: {data_dir} got empty csv data files"

    def _init_var(self):
        self._csv_idx = self._rng_csv_idx.randint(0, len(self._csv_list)-1)
        self._timestep = 0

        # read csv
        csv_file = self._csv_list[self._csv_idx]
        assert isfile(csv_file), f"{csv_file} is not a file"
        _df = pd.read_csv(csv_file)
        assert _df.shape[0]>0, f"{csv_file} got empty df data"
        #proccess
        _df = standardlize_df(_df)
        self._df = fill_missing_frame(_df)# filling missing frame
        date = self._df.index[0]
        _s = self._set_clock_func(date, 9, 30)
        _e = self._set_clock_func(date, 15, 59)
        self._df = self._df.truncate(before=_s, after=_e)

        for stat in self._stat_keys:
            self._df[stat] = np.nan # create empty comlumn
        self._df["action"] = np.nan
        
        self._df['stat_balance'].iloc[self._timestep] = self._init_balance
        self._df['stat_profit_ratio'].iloc[self._timestep] = 0
        self._df['stat_pnl'].iloc[self._timestep] = 0
    



    def _get_obs(self):
        obs = {}
        for k in self._obs_keys:
            obs[k] = self._df.iloc[self._timestep][k]
        return obs
    

    def _update_stats(self, action):
        self._df['action'].iloc[self._timestep-1] = action 
        close = self._df['close'].iloc[self._timestep]

        action_prv = 0 if self.timestep == 1 else self._df['stat_action'].iloc[self._timestep-1]
        sell_price = self._df['open'].iloc[self._timestep]
        profit_ratio = (sell_price - self.buy_price) / self.buy_price 
        balance =  self._buy_pos*(profit_ratio-2*self._commision_rate) + self.buy_balance



        # if is_hold_prv and (not is_hold): #sell
        #     sell_price = self._df['open'].iloc[self._timestep]
        #     profit_ratio = (sell_price - self.buy_price) / self.buy_price 
        #     balance =  self._buy_pos*(profit_ratio-2*self._commision_rate) + self.buy_balance
        # elif is_hold_prv and is_hold:# hold
        #     self.buy_init_balance = self._df['stat_balance'].iloc[self._timestep]
        #     profit_ratio = (close - self.buy_price) / self.buy_price
        #     balance =  self._buy_pos*(profit_ratio-self._commision_rate) + self.buy_balance
        # elif (not is_hold_prv) and is_hold: # buy
        #     self.buy_price = self._df['open'].iloc[self._timestep]
        #     self.buy_balance = balance_prv
        #     profit_ratio = (close - self.buy_price) / self.buy_price
        #     balance =  self._buy_pos*(profit_ratio-self._commision_rate) + self.buy_balance
        # else: # not holding 
        #     self.buy_price = None
        #     self.buy_balance = None
        #     profit_ratio =0 
        #     balance = balance_prv

        # print(commission_rate)
        self._df['stat_profit_ratio'].iloc[self._timestep] =  profit_ratio
        self._df['stat_balance'].iloc[self._timestep] = balance
        self._df['stat_pnl'].iloc[self._timestep] =  profit_ratio * self._buy_pos

        
    @property
    def df(self):
        return self._df

    @property
    def timestep(self):
        return self._timestep

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(1,)) # delta action, 0~1 buy, -1~0 sell, 0 hold

    @property
    def observation_space(self):
        obs = {}
        for v in self._obs_keys:
            if v=="stat_position_ratio":
                obs[v] = gym.spaces.Box(low=0,high=1,shape=(1,),dtype=np.float)
            else:
                raise NotImplementedError
        return gym.spaces.Dict(obs)
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng_csv_idx = np.random.RandomState(seed) 