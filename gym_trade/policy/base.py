from abc import ABC, abstractmethod


class BasePolicy(ABC):
    def __init__(self):
        pass

    
    @abstractmethod
    def __call__(self, obs, **kwargs):
        action = 0
        return action
        
    @abstractmethod
    def init_policy(self,**kwargs):
        pass

    @property 
    @abstractmethod
    def obs_keys(self):
        """
        A list of keys that env need to return
        """
        return []

