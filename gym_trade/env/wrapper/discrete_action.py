from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import gym
class DiscreteAction(BaseWrapper):
    def __init__(self, env, 
                 action_scale=0.2,
                 **kwargs):
        super().__init__(env)
        self._dis = [1.0, -1.0, 0.0] # buy, sell, hold
        self._action_scale = action_scale

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._dis))
    
    def _dis2con(self, dis): # discrete to continuous action
        return self._dis[dis] * self._action_scale
    
    def step(self, action):
        action = self._dis2con(action)
        return self.env.step(np.array([action]))
    
    @property
    def hold_action(self,):
        return 2
    
    # def get_oracle_action(self):
    #     pass