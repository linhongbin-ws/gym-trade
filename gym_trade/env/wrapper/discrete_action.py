from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
class DiscreteAction(BaseWrapper):
    def __init__(self, env, 
                 action_scale=0.2,
                 **kwargs):
        super().__init__(env)
        self._dis = {
            "buy": np.array([1.0]),
            "sell": np.array([-1.0]),
            "hold": np.array([0.0]),
        }
        self._dis_name = ["buy", "sell", "hold"]
    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._dis))
    
    def _dis2con(self, dis):
        return self._dis[self._dis_name[dis]]
    
    def step(self, action):
        action = self_dis2con(action)
        return self.env.step(action)
    
    
    # def get_oracle_action(self):
    #     pass