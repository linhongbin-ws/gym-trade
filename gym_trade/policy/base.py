from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from copy import deepcopy
class BasePolicy(ABC):
    def __init__(self, hyper_param_range: dict[str, Any]):
        for k, v in hyper_param_range.items():
            assert len(v) ==3, f"hyper_param_range should be (value, min, max)"
            assert v[0] >= v[1] and v[0] <= v[2], f"(value, min, max), value should be in range [{v[1]}, {v[2]}]"
        self._hyper_param_range = hyper_param_range


    def randomize_hyper_param(self, random_type: str | None = None):
        if random_type is None:
            return {k: v[0] for k, v in self._hyper_param_range.items()}
        elif random_type == "uniform":
            return {k: np.random.uniform(v[1], v[2]) for k, v in self._hyper_param_range.items()}
        else:
            raise ValueError(f"random_type {random_type} not supported")

    def set_hyper_param(self, hyper_param: dict[str, Any]):
        self.hyper_param = deepcopy(hyper_param)

        
    def init_policy(self,**kwargs):
        pass

    @abstractmethod
    def __call__(self, obs, **kwargs):
        action = 0
        return action
    
    @property
    @abstractmethod
    def obs_keys(self):
        """this give a list of obs key of this policy"""
        return []