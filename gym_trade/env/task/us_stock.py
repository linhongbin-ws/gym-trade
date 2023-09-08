import gym
from gym import spaces
import pandas as pd
import numpy as np
from os.path import isfile
import random
from typing import List, Union


import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
pd.plotting.register_matplotlib_converters() # fix bug in windows for pyplot
matplotlib.use('TKAgg')
import mplfinance as mpf
pd.options.mode.chained_assignment = None  # disable warning for pandas

#=== gym_datatrade package
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from gym_trade.tool import ta
from gym_trade.tool.common import get_csv_dir


class US_Stock_Env(gym.Env):
    def __init__(self, data_dir, 
                    feature_args={'open':None, 'high':None,'low':None,'close':None},
                    initial_account_balance=100,
                    buy_position = 100,
                    commission_rate = 3/1e4, 
                    reward_type='pnl',
                    reward_is_sparse=True,
                    obs_keys=["open", "high", "low", "close", "is_hold"],
                    seed=0,
                    verbose=0,
                    ):
        
        self.INITIAL_ACCOUNT_BALANCE = initial_account_balance
        self.BUY_POSITION = buy_position
        self.COMMISSION_RATE = commission_rate
        self.reward_type = reward_type
        self.reward_is_sparse = reward_is_sparse
        self.feature_args = feature_args
        self.obs_keys = obs_keys
        self.set_clock = lambda date, hour, minute: date.replace(hour=hour, minute=minute)
        self.verbose = verbose
        
        if isinstance(data_dir, str): # extract csv file path
            self.csv_list = get_csv_dir(data_dir)
        elif isinstance(csv_dir, list):
            self.csv_list = csv_dir
        else:
            raise NotImplementedError
        assert len(self.csv_list)!=0, f"data_dir: {data_dir} got empty csv data files"

        self.seed = seed
        _ = self.reset()
        
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int)
        self._seed = seed
        self.rng_csv_idx = np.random.RandomState(seed) 

    def reset(self):
        #========= init var=====
        self._csv_idx = self.rng_csv_idx.randint(0, len(self.csv_list)-1)
        self._current_timestep = 0
        self._position = 0
        self._balance = self.INITIAL_ACCOUNT_BALANCE
        self.feature_names = []

        #========= dataframe ========
        csv_file = self.csv_list[self._csv_idx]
        assert isfile(csv_file), f"{csv_file} is not a file"
        _df = pd.read_csv(csv_file)
        assert _df.shape[0]>0, f"{csv_file} got empty df data"
        _df = standardlize_df(_df)
        self._df = fill_missing_frame(_df)# filling missing frame
        date = self._df.index[0]
        _s = self.set_clock(date, 9, 30)
        _e = self.set_clock(date, 15, 59)
        self._df = self._df.truncate(before=_s, after=_e)
        stat_names=['is_hold', 'balance', 'profit_ratio','is_action', 'action', 'pnl']
        for stat in stat_names:
            self._df[stat] = np.nan # create empty comlumn
        self._df['is_hold'].iloc[self._current_timestep] = False
        self._df['balance'].iloc[self._current_timestep] = self.INITIAL_ACCOUNT_BALANCE
        self._df['profit_ratio'].iloc[self._current_timestep] = 0
        self._df['time_index'] = [i for i in range(len(self._df.index))]
        self._create_df_feature() # create feature

        obs = self._get_obs()
        return obs
    


    def step(self, action):
        self._current_timestep +=1
        assert self._current_timestep <=self._df.shape[0] - 1
        is_hold_prv = self._df['is_hold'].iloc[self._current_timestep-1] 
        if action == 0: # toggle action
            is_hold = not is_hold_prv
            is_action =True
        elif action == 1: # idle action
            is_hold = is_hold_prv
            is_action =False
        else:
            raise NotImplementedError
        self._df['is_hold'].iloc[self._current_timestep] = is_hold
        self._df['is_action'].iloc[self._current_timestep-1] = is_action
        self._df['action'].iloc[self._current_timestep-1] = action
        self._update_stats() 

        #======= outputs========
        info = self._get_info()
        obs = self._get_obs()
        done = len(info['done']) !=0 
        reward = 0
        if is_action:
            reward -= self.COMMISSION_RATE 
        if is_hold_prv and (not is_hold): #sell
            reward += info['profit_ratio'] 
        
        return obs, reward, done, info




    def render(self, 
                mode = 'human', 
                plot_features_dict={}):
        assert mode == 'human', f'{mode} is not support'

        _plot_features_dict = {feature_name:None for feature_name in self.feature_names}
        _plot_features_dict.update(plot_features_dict)
        style.use('ggplot')
        _df = self._df.iloc[:min(self._current_timestep+1, self._df.shape[0])].copy()
        df_mpf = _df[['open','high','low','close','volume']].copy()
        df_mpf = df_mpf.rename(columns={"open": "Open",
                                        "high": "High",
                                        "low": "Low",
                                        "close": "Close",
                                        "volume": "Volume"}) # names for mplfinance
        df_mpf.index.name = 'Date'
        panel_idx = 1
        ap =[]
        ts = min(self._current_timestep+1, self._df.shape[0])
        buy_actions = self._df['close'].iloc[:ts].copy() *0.98
        sell_actions = self._df['close'].iloc[:ts].copy() *1.02
        # print(self._df['is_hold'].iloc[:ts]==True)
        buy_actions[~((self._df['is_hold'].iloc[:ts]==True) & (self._df['is_hold'].shift(1).iloc[:ts]==False))] = np.nan
        sell_actions[~((self._df['is_hold'].iloc[:ts]==False) & (self._df['is_hold'].shift(1).iloc[:ts]==True))] = np.nan
        buy_actions = buy_actions.to_list()
        sell_actions = sell_actions.to_list()
        # print(np.sum(buy_actions))
        if not np.isnan(buy_actions).all():
            ap.append(mpf.make_addplot(buy_actions, type='scatter', panel=0,
                                color = 'red', markersize=50, marker='^'))
        if not np.isnan(sell_actions).all():
            ap.append(mpf.make_addplot(sell_actions, type='scatter', panel=0,
                                color = 'green', markersize=50, marker='v'))    
        for k, v in _plot_features_dict.items():
            if v is None:
                panel_idx +=1
                _panel_idx = panel_idx
            else:
                _panel_idx = v

            ap.append(mpf.make_addplot(self._df[k].iloc[:ts], panel=_panel_idx,
                                    type='line', ylabel=k))
        mpf.plot(df_mpf, addplot = ap, volume = True, type='candle',mav=(3), title=self.csv_list[self._csv_idx])

    @property
    def timestep(self):
        return self._current_timestep

    @property
    def action_space(self):
        return gym.spaces.Discrete(2) # Toggle_hold, Idle

    @property
    def observation_space(self):
        obs = {}
        obs['scalers'] = gym.spaces.Box(low=-np.Inf,
                            high=np.Inf,
                            shape=(len(self.feature_names), ),
                            dtype=np.float) 
        return gym.spaces.Dict(obs)

    def _create_df_feature(self):
        self.feature_names = []
        for k, v in self.feature_args.items():
            if k not in self._df.columns:
                _call = getattr(ta, k)
                # assert feature in self.feature_args
                args = [self._df]
                args.extend(v.copy())
                self._df, feature_names = _call(*args)
                self.feature_names.extend(feature_names)
            else:
                self.feature_names.append(k)

    def _get_obs(self):
        obs = {}
        for k in self.obs_keys:
            obs[k] = self._df.iloc[:self._current_timestep][k]
            obs[k] = obs[k].to_numpy() if isinstance(obs[k], pd.Series) else np.array([obs[k]])
            obs[k] = np.concatenate([np.zeros(self._df.shape[0]-obs[k].shape[0], dtype=float),
                                    obs[k].astype(float)], axis=0) # by default padded with zeros
        return obs
    

    def _update_stats(self):
        is_hold_prv = self._df['is_hold'].iloc[self._current_timestep-1] 
        balance_prv = self._df['balance'].iloc[self._current_timestep-1]
        is_hold = self._df['is_hold'].iloc[self._current_timestep]
        close = self._df['close'].iloc[self._current_timestep]

        
        if is_hold_prv and (not is_hold): #sell
            sell_price = self._df['open'].iloc[self._current_timestep]
            profit_ratio = (sell_price - self.buy_price) / self.buy_price 
            balance =  self.BUY_POSITION*(profit_ratio-2*self.COMMISSION_RATE) + self.buy_balance
        elif is_hold_prv and is_hold:# hold
            self.buy_init_balance = self._df['balance'].iloc[self._current_timestep]
            profit_ratio = (close - self.buy_price) / self.buy_price
            balance =  self.BUY_POSITION*(profit_ratio-self.COMMISSION_RATE) + self.buy_balance
        elif (not is_hold_prv) and is_hold: # buy
            self.buy_price = self._df['open'].iloc[self._current_timestep]
            self.buy_balance = balance_prv
            profit_ratio = (close - self.buy_price) / self.buy_price
            balance =  self.BUY_POSITION*(profit_ratio-self.COMMISSION_RATE) + self.buy_balance
        else: # not holding 
            self.buy_price = None
            self.buy_balance = None
            profit_ratio =0 
            balance = balance_prv

        # print(commission_rate)
        self._df['profit_ratio'].iloc[self._current_timestep] =  profit_ratio
        self._df['balance'].iloc[self._current_timestep] = balance
        self._df['pnl'].iloc[self._current_timestep] =  profit_ratio * self.BUY_POSITION


    def _get_info(self):
        reward = 0
        info = {'done':[],
                'profit_ratio':0}

        if self._current_timestep >=self._df.shape[0]-1:
            reward += (self._df['balance'].iloc[self._current_timestep] 
                        - self.INITIAL_ACCOUNT_BALANCE)/self.INITIAL_ACCOUNT_BALANCE
            _info = {'time limit': None}
            info['done'].append(_info)
        
        info['profit_ratio'] = self._df['profit_ratio'].iloc[self._current_timestep]
        return info