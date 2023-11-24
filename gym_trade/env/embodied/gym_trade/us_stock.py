import gym
from gym import spaces
import pandas as pd
import numpy as np
from os.path import isfile
import random
from typing import List, Union
import pathlib


#=== gym_datatrade package
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from gym_trade.tool import ta
from gym_trade.tool.common import get_csv_dir


class US_Stock_Env(gym.Env):
    def __init__(self, 
                    csv_root_dir='', 
                    init_balance=100,
                    commission_type = "futu", 
                    reward_type='pnl_delta_dense',
                    obs_keys=["stat_posRate"],
                    stat_keys=['stat_pos', 'stat_posRate', 'stat_pnl','stat_balance','stat_cash',],
                    action_min_thres=0.1,
                    **kwargs,
                    ):
        self._csv_root_dir = str(pathlib.Path( __file__ ).absolute().parent.parent.parent / "asset" / "mini_minute_data")  if csv_root_dir=='' else  csv_root_dir
        self._init_balance = init_balance
        self._commission_type = commission_type
        self._reward_type = reward_type
        self._obs_keys = obs_keys
        self._stat_keys = stat_keys
        self._set_clock_func = lambda date, hour, minute: date.replace(hour=hour, minute=minute)
        self._action_min_thres = action_min_thres
        
        self. _update_csv_dir(self._csv_root_dir)
        
        self._seed = 0
        self.seed = 0
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
        info = {}
        obs = self._get_obs()
        done = self._fsm()
        reward = self._get_reward() 

        return obs, reward, done, info

    def _get_reward(self):
        if self._reward_type == "pnl_delta_dense":
            pnl_delta = self._df['stat_pnl'].iloc[self._timestep]-self._df['stat_pnl'].iloc[self._timestep-1]
            return pnl_delta
        else:
            raise NotImplementedError
    def _fsm(self):
        done = self.timestep >= (len(self._df.index)-1) 
        return done

    def _update_csv_dir(self, dir):
        if isinstance(dir, str): # extract csv file path
            self._csv_list = get_csv_dir(dir)
        # elif isinstance(csv_dir, list):
        #     self._csv_list = csv_dir
        else:
            raise NotImplementedError
        assert len(self._csv_list)!=0, f"data_dir: {dir} got empty csv data files"

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
        self._df['stat_cash'].iloc[self._timestep] = self._init_balance
        self._df['stat_pos'].iloc[self._timestep] = 0
        self._df['stat_posRate'].iloc[self._timestep] = 0
        self._df['stat_pnl'].iloc[self._timestep] = 0
    


    def _get_obs(self):
        obs = {}
        for k in self._obs_keys:
            obs[k] = self._df.iloc[self._timestep][k]
        return obs
    
    def _get_commision(self, pos):
        if self._commission_type == "futu":
            min_commision = 0.99
            commision = 0.0049
            platform = 0.005
            min_platform = 1
            return np.clip(pos*commision,min_commision,None) + np.clip(pos*platform,min_platform,None) 
        else:
            raise NotImplementedError
    

    def _update_stats(self, action):
        self._df['action'].iloc[self._timestep-1] = action 
        _close = self._df['close'].iloc[self._timestep]
        _open = self._df['open'].iloc[self._timestep]
        _cash_prv = self._df['stat_cash'].iloc[self._timestep-1]
        _pos_prv = self._df['stat_pos'].iloc[self._timestep-1]
        

        if action>=self._action_min_thres:
            _buy_price = _open
            _buy_cash = _cash_prv * action 
            _buy_pos = _buy_cash // _open
            _buy_cash = _buy_pos * _open
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv - _buy_cash - self._get_commision(_buy_pos)
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv + _buy_pos 
        elif action<=-self._action_min_thres:
            _sell_price = _open
            _sell_pos = np.floor(_pos_prv * np.abs(action))
            _sell_cash = _sell_pos * _sell_price
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv + _sell_cash - self._get_commision(_sell_pos) 
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv - _sell_pos
        else:
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv

        self._df['stat_balance'].iloc[self._timestep] = self._df['stat_pos'].iloc[self._timestep] * _close  + \
                                                        self._df['stat_cash'].iloc[self._timestep]

        self._df['stat_pnl'].iloc[self._timestep] = (self._df['stat_balance'].iloc[self._timestep] - self._init_balance)\
                                                                 / self._init_balance 
        self._df['stat_posRate'].iloc[self._timestep] = (self._df['stat_balance'].iloc[self._timestep] - self._df['stat_cash'].iloc[self._timestep])\
                                                             / self._df['stat_balance'].iloc[self._timestep]
    def render(self):
        raise NotImplementedError
        
    @property
    def df(self):
        return self._df

    @property
    def timestep(self):
        return self._timestep

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(1,)) # delta action, thres~1: buy, -1~-thres: sell, -thres~thres: hold

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

    @property
    def timestep(self):
        return self._timestep 

    @property
    def stat_keys(self):
        return self._stat_keys
    @property
    def obs_keys(self):
        return self._obs_keys