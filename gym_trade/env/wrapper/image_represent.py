from gym_trade.env.wrapper.base import BaseWrapper
import numpy as np
import cv2
from gym_trade.tool.common import scale_arr

class ImageRepresent(BaseWrapper):

    def __init__(self, 
                    env,
                    method='simple',
                    time_window=64,
                    value_window=64,
                    **kwargs
                 ):
        super().__init__(env,)
        self._method = method
        self._time_window = time_window
        self._value_window = value_window


    def render(self,):
        ohlcv_mat = self.unwrapped.df[['open', 'high', 'low', 'close','volume']].iloc[:self.unwrapped.timestep+1].to_numpy()
        img = self._get_image(ohlcv_mat)
        imgs = {"rgb": img}
        return imgs

    def _get_image(self, ohlcv_mat):
        
        # print(self.unwrapped.timestep)
        # print(ohlcv_mat)
        if self._method == "simple":
            _s = ohlcv_mat.shape
            _ohlcv_mat = ohlcv_mat[max(0, _s[0]-self._time_window):,:] # truncated matrix 

            _ohlc_mat = _ohlcv_mat[:,:4]
            _ohlc_mat = scale_arr(_ohlc_mat, np.min(_ohlc_mat),
                                            np.max(_ohlc_mat), 
                                            0,
                                            +self._value_window-1,) # scale to [0 ,value_window-1] range

            _ohlc_mat = np.uint8(_ohlc_mat)
            candle_mat = np.zeros((self._value_window,self._time_window,), dtype=np.uint8)
            _v_id_o = _ohlc_mat[:,0].tolist()
            _v_id_c = _ohlc_mat[:,3].tolist()
            _t_id = [i for i in range(len(_v_id_c))]
            for j in range(len(_t_id)):
                _low = min(_v_id_o[j], _v_id_c[j])
                _high = max(_v_id_o[j], _v_id_c[j])
                candle_mat[_low:_high+1,_t_id[j]] = 255
            layer1 = candle_mat
            img_mat = np.stack([layer1]*3, axis=2)
        return img_mat
            
            