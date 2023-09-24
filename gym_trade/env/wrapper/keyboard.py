from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import gym
from pynput import keyboard
class Keyboard(BaseWrapper):
    def __init__(self, env, 
                 **kwargs):
        super().__init__(env)

    def _keyboard(self):
        # The event listener will be running in this block
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Release):
                    break
        return event.key.char if hasattr(event.key, "char") else event.key

    def reset(self):
        obs = self.env.reset()
        # self._keyboard()
        return obs
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        c = self._keyboard()
        info.update({"keyboard": c})
            
        return obs, reward, done, info
    
