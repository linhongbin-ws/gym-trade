from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import cv2

class DSA(BaseWrapper):

    def __init__(self, 
                    env,
                    **kwargs
                 ):
        super().__init__(env,)


    def render(self,):
        ohlc_mat = self.unwrapped.df[['open', 'high', 'low', 'close']]
        img = self._get_image(ohlc_mat)

        return img

    def _get_image(ohlc_mat):