from abc import ABC, abstractmethod


class BasePolicy(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def __call__(self, obs, **kwargs):
        action = 0
        return action
    
    def init_policy(self,**kwargs):
        pass



        