from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import gym
from gym_trade.tool import ta
class TA(BaseWrapper):
    def __init__(self, env,
                 **kwargs):
        super().__init__(env)
        # print(ta_list)
        self._ta_dict = {}
        for k,v in kwargs.items():
            dot = k.find(".")
            ta_name = k[:dot]
            arg_name = k[dot+1:]
            if not (ta_name in self._ta_dict):
                self._ta_dict[ta_name] = {"func": None, "args": {}}
            if arg_name == "func":
                self._ta_dict[ta_name][arg_name] = v
            else:
                self._ta_dict[ta_name]["args"][arg_name] = v


    def reset(self):
        obs = self.env.reset()
        self._create_ta()
        obs.update(self._get_obs_from_df())
        return obs
    
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        obs.update(self._get_obs_from_df())
        return obs, reward, done, info
    

    def _get_obs_from_df(self):
        df = self.unwrapped.df
        obs = {}
        for k, v in self._ta_dict.items():
            obs[k]  = df[k].iloc[self.unwrapped.timestep-1]
            # print(obs[k],self.unwrapped.timestep)
        return obs


    def _create_ta(self):
        df = self.unwrapped.df
        obs = {}
        for k, v in self._ta_dict.items():
            _f_name = v["func"]
            _call = getattr(ta, _f_name)
            _args = v["args"]
            _args['df'] = df
            df[k] = _call(**_args)
        
        self.unwrapped.df = df


