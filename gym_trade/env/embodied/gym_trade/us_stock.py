import gym
import pandas as pd
import numpy as np
from os.path import isfile
from typing import List, Union
import pathlib

## gym_datatrade package
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
# from gym_trade.tool.common import get_csv_dir


class US_Stock_Env(gym.Env):
    def __init__(self, 
                    df_list, # list of csv file names or list of df 
                    init_balance=100,
                    commission_type = "futu", 
                    reward_type='pnl_delta_sparse',
                    obs_keys=["position_ratio","open","close","high","low"],
                    stat_keys=['stat_pos', 'position_ratio', 'stat_pnl','stat_balance','stat_cash',],
                    action_min_thres=0.1,
                    fix_buy_position=True,
                    interval="minute",
                    action_on_prv_close=True,
                    **kwargs,
                    ):
        self._init_balance = init_balance
        self._commission_type = commission_type
        self._reward_type = reward_type
        self._obs_keys = obs_keys
        self._stat_keys = stat_keys
        self._set_clock_func = lambda date, hour, minute: date.replace(hour=hour, minute=minute)
        self._action_min_thres = action_min_thres
        self._fix_buy_position = fix_buy_position
        self._interval = interval
        self._action_on_prv_close = action_on_prv_close
        assert self._interval in ["day", "minute"]
        self.seed = 0
        # self. _update_csv_dir(_csv_root_dir) # load csv directory
        self._df_list = df_list
        self._init_var() # init variables

    #====== gym api ========
    def reset(self):
        self._init_var()
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        # action clip to -1 to 1
        action = np.clip(action,-1,1)

        self._timestep +=1
        assert self._timestep <=self._df.shape[0] - 1
        self._update_stats(action) 
        self._update_reward(action)
        info = {}
        obs = self._get_obs()
        done = self._fsm()
        return obs, self._reward, done, info
    
    def render(self):
        raise NotImplementedError

    @property
    def action_space(self):  
        # delta action, thres~1: buy, -1~-thres: sell, -thres~thres: hold=0
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
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng_csv_idx = np.random.RandomState(seed)

    @property
    def action_space(self):
        return gym.spaces.Box(-1,1)
    
    #====== other ==============
    def _update_stats(self, action):
        self._df['action'].iloc[self._timestep-1] = action 
        _close = self._df['close'].iloc[self._timestep]
        _open = self._df['open'].iloc[self._timestep]
        _cash_prv = self._df['stat_cash'].iloc[self._timestep-1]
        _pos_prv = self._df['stat_pos'].iloc[self._timestep-1]

        if self._action_on_prv_close:
            action_price = self._df['close'].iloc[self._timestep-1]
        else:
            action_price = _open 
        # buy
        if action>=self._action_min_thres:
            _buy_cash = _cash_prv * action  # proportion to action value, 0.8 to save some money for commision
            _buy_pos = _buy_cash // action_price # buy at open point
            _buy_cash = _buy_pos * action_price
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv - _buy_cash - self._get_commision(_buy_pos)
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv + _buy_pos 
        
        # sell
        elif action<=-self._action_min_thres:
            _sell_price = action_price
            _sell_pos = np.floor(_pos_prv * np.abs(action))
            _sell_cash = _sell_pos * _sell_price
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv + _sell_cash - self._get_commision(_sell_pos) 
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv - _sell_pos
        
        # hold
        else:
            self._df['stat_cash'].iloc[self._timestep] = _cash_prv
            self._df['stat_pos'].iloc[self._timestep] = _pos_prv
        self._df['stat_balance'].iloc[self._timestep] = self._df['stat_pos'].iloc[self._timestep] * _close  + \
                                                        self._df['stat_cash'].iloc[self._timestep]
        self._df['stat_pnl'].iloc[self._timestep] = (self._df['stat_balance'].iloc[self._timestep] - self._init_balance)\
                                                                 / self._init_balance 
        self._df['position_ratio'].iloc[self._timestep] = (self._df['stat_balance'].iloc[self._timestep] - self._df['stat_cash'].iloc[self._timestep])\
                                                             / self._df['stat_balance'].iloc[self._timestep]

    def _update_reward(self, action):
        sparse_update_condition = self._df['position_ratio'].iloc[self._timestep]<=0 and (self._df['position_ratio'].iloc[self._timestep-1]>0)
        if sparse_update_condition and self._reward_type == "pnl_delta_sparse":
            self._reward = 0
        else:
            self._reward =  self._df['stat_pnl'].iloc[self._timestep]-self._stat_pnl_prv
        if (sparse_update_condition and self._reward_type == "pnl_delta_sparse") or self._reward_type == "pnl_delta_dense":
            self._stat_pnl_prv = self._df['stat_pnl'].iloc[self._timestep]

    def _update_csv_dir(self, dir_or_filelist):
        if isinstance(dir_or_filelist, str): 
            # extract csv file path
            self._df_list = get_csv_dir(dir_or_filelist)
        elif isinstance(dir_or_filelist, list):    
            self._df_list = dir_or_filelist
        else:
            raise NotImplementedError
        assert len(self._df_list)!=0, f"data_dir: {dir} got empty csv data files"

    def _fsm(self):
        done = self.timestep >= (len(self._df.index)-1) 
        return done
    
    def _init_var(self):
        # init
        self._csv_idx = 0 if len(self._df_list)==1 else self._rng_csv_idx.randint(0, len(self._df_list)-1) 
        self._timestep = 0

        # read csv
        csv_file = self._df_list[self._csv_idx]
        if isinstance(csv_file, str):
            self._csv_name = csv_file
            # assert isfile(csv_file), f"{csv_file} is not a file"
            _df = pd.read_csv(csv_file)
        else:
            _df = csv_file
        assert _df.shape[0]>0, f"{csv_file} got empty df data"

        #proccess

        if self._interval == "minute":
            self._df = standardlize_df(_df) 
            self._df = fill_missing_frame(self._df)# filling missing frame
            date = self._df.index[0]
            _s = self._set_clock_func(date, 9, 30)
            _e = self._set_clock_func(date, 15, 59)
            self._df = self._df.truncate(before=_s, after=_e)
        else:
            _df.rename(columns={"Date": "datetime",},inplace=True)
            # print(_df.columns)
            self._df = standardlize_df(_df) 
        for stat in self._stat_keys:
            self._df[stat] = np.nan # create empty comlumn
        self._df["action"] = np.nan
        self._df['stat_balance'].iloc[self._timestep] = self._init_balance
        self._df['stat_cash'].iloc[self._timestep] = self._init_balance
        self._df['stat_pos'].iloc[self._timestep] = 0
        self._df['position_ratio'].iloc[self._timestep] = 0
        self._df['stat_pnl'].iloc[self._timestep] = 0
        self._stat_pnl_prv = 0
        self._reward = 0
        # print(self._df.columns)
    def _get_obs(self):
        obs = {}
        for k in self._obs_keys:
            if k=="timestep":
                obs[k] = self.timestep
            else:
                obs[k] = self._df.iloc[self._timestep][k]
        return obs
    
    def _get_commision(self, pos):
        if self._commission_type == "futu":
            min_commision = 0.99
            commision = 0.0049
            platform = 0.005
            min_platform = 1
            return np.clip(pos*commision,min_commision,None) + np.clip(pos*platform,min_platform,None) 
        elif self._commission_type == "free":
            return 0
        else:
            raise NotImplementedError

    @property
    def df(self):
        return self._df
    

    
    @property
    def timestep(self):
        return self._timestep

    
    @property
    def file(self):
        name = pathlib.Path(self._csv_name)
        return name.stem
    
    @property
    def pnl(self):
        pnl = self.df.iloc[self.timestep]["stat_pnl"]
        return pnl
    
    # @property
    # def stat_keys(self):
    #     return self._stat_keys
    # @property
    # def obs_keys(self):
    #     return self._obs_keys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=str, default="minute")
    parser.add_argument('--csvdir', type=str, default="./gym_trade/asset/mini_minute_data/")
    parser.add_argument('--action', type=str, default="random") #"random" "bnh"(buy and hold)
    parser.add_argument('--vis', action="store_true") #"random" "bnh"(buy and hold)
    args = parser.parse_args()
    env = US_Stock_Env(csv_root_dir=args.csvdir,interval=args.interval)
    if args.vis:
        from gym_trade.env.wrapper import LightChart_Visualizer
        env =  LightChart_Visualizer(env, subchart_keys=[], keyboard=True)
    env.reset()
    done = False
    sum_reward = 0
    while not done:
        if args.action == "random":
            action = env.action_space.sample()
        elif args.action == "bnh":
            if env.timestep == 1:
                action = 1
            else:
                action = 0
        obs, reward, done, info = env.step(action)
        sum_reward+=env.pnl
        # print(env.df)
        if args.vis:
            env.gui_show()
    print("stock: ", env.file)
    print("pnl: ", env.pnl)