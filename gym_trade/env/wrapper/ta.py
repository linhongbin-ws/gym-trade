from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import gym
from gym_trade.tool import ta
class TA(BaseWrapper):
    def __init__(self, env,
                 ta_args={"ma": {"func": "ma", "key":"open", "window": 3}},
                 **kwargs):
        super().__init__(env)
        self._ta_args = ta_args

    def reset(self):
        obs = self.env.reset()
        _obs = self._process()
        obs.update(_obs)
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        _obs = self._process()
        obs.update(_obs)
        return obs, reward, done, info

    def _process(self):
        df = self.unwrapped.df[['open', 'high', 'low', 'close','volume']].iloc[:self.unwrapped.timestep+1]
        obs = {}
        for k, v in self._ta_args.items():
            _f_name = v["func"]
            _call = getattr(ta, _f_name)
            _kwargs =v.copy()
            _kwargs.pop("func", 0)
            seri = _call(df, **_kwargs)
            df[k] = seri
            obs[k] = seri.iloc[-1]
        return obs

