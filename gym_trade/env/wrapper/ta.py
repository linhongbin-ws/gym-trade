from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
from gym_trade.tool import ta

class TA(BaseWrapper):
    def __init__(self, env,
                 ta_dict_list,
                 **kwargs):
        super().__init__(env)
        self._ta_dict_list = ta_dict_list


    def reset(self):
        obs = self.env.reset()
        self._create_ta()
        obs.update(self._get_obs_from_df())
        return obs
    
    def step(self,action):
        _action = action
        # if 'trade_curb' in self.unwrapped.df:
        #     if self.unwrapped.df['trade_curb'].iloc[self.unwrapped.timestep+1]:
        #         _action = self.env.hold_action                
        obs, reward, done, info = self.env.step(_action)
        obs.update(self._get_obs_from_df())
        return obs, reward, done, info
    

    def _get_obs_from_df(self):
        df = self.unwrapped.df
        obs = {}
        for v in self._ta_col_names:
            obs[v]  = df[v].iloc[self.unwrapped.timestep]
            # print(obs[k],self.unwrapped.timestep)
        return obs


    def _create_ta(self):
        df = self.unwrapped.df.copy()
        
        ta_col_names = []
        for ta_dict in self._ta_dict_list:
            f_name = None
            args = {'df':df}
            for k,v in ta_dict.items():
                if k == 'func':
                    f_name = v
                else:
                    args[k] = v
            assert f_name != None


            call = getattr(ta, f_name)
            ret = call(**args)

            for ret_k, ret_v in ret.items():
                df[ret_k] = ret_v
                ta_col_names.append(ret_k)

        self._ta_col_names = ta_col_names
        self.unwrapped.df = df


