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
                    reverse_up_down_color=True,
                    **kwargs
                 ):
        super().__init__(env,)
        self._method = method
        self._time_window = time_window
        self._value_window = value_window
        self._reverse_up_down_color = reverse_up_down_color


    def render(self,):
        ohlcv_mat = self.unwrapped.df[['open', 'high', 'low', 'close','volume']].iloc[:self.unwrapped.timestep+1].to_numpy()
        
        img = self._get_image(ohlcv_mat)
        imgs = {"rgb": img}
        return imgs

    def _get_image(self, _ohlcv_mat_):
        ohlcv_mat = _ohlcv_mat_.copy()
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

            tuple_list = [(1,2,128),(0,3,255)]
            for v in tuple_list:
                _v_id_o = _ohlc_mat[:,v[0]].tolist()
                _v_id_c = _ohlc_mat[:,v[1]].tolist()
                _t_id = [i for i in range(len(_v_id_c))]
                # print("fill ",len(_t_id))
                for j in range(len(_t_id)):
                    _low = min(_v_id_o[j], _v_id_c[j])
                    _high = max(_v_id_o[j], _v_id_c[j])
                    candle_mat[_low:_high+1,_t_id[j]] = v[2] # fill color to grids
            layer1 = candle_mat

            
            _z = np.zeros((self._value_window - _ohlc_mat.shape[0],4))

            _ohlc_mat_full = np.concatenate((_ohlc_mat,  _z), axis=0)
            up_bool = _ohlc_mat_full[:,3] > _ohlc_mat_full[:,0]
            # print(up_bool)
            up_bools = np.stack([up_bool]*self._value_window, axis=0)
            down_bools = np.logical_not(up_bools)
            layer1 = np.zeros((self._value_window,self._time_window,), dtype=np.uint8)
            layer2 = layer1.copy()
            layer3 = layer1.copy()
            layer1[up_bools] = candle_mat[up_bools]
            layer2[down_bools] = candle_mat[down_bools]
            
            img_list = [layer2, layer1,layer3]if self._reverse_up_down_color else [layer1,layer2,layer3]
            
            img_mat = np.stack(img_list, axis=2)

            img_mat = np.flip(img_mat, axis=0) # sync with human view
        
        elif self._method == "donchain":
            if ohlcv_mat.shape[0]-self._time_window < 0:
                first_vec = ohlcv_mat[0,:]
                first_mat = np.stack([first_vec]*(self._time_window-ohlcv_mat.shape[0]), axis=0)
                ohlcv_mat = np.concatenate((first_mat, ohlcv_mat),axis=0)
            ohlcv_mat = ohlcv_mat[-self._time_window:,:]

            hl_mat = ohlcv_mat[:,[1,2]]
            hl_mat = scale_arr(hl_mat, np.min(hl_mat),
                                            np.max(hl_mat), 
                                            0,
                                            +self._value_window-1,) # scale to [0 ,value_window-1] range

            hl_mat = np.uint8(hl_mat)
            fill_mat = np.zeros((self._value_window,self._time_window,), dtype=np.uint8)
            tuple_list = [(0,1,255)]
            for v in tuple_list:
                high_id = hl_mat[:,v[0]].tolist() # high
                low_id = hl_mat[:,v[1]].tolist() # low
                for j in range(len(high_id)):
                    _low = low_id[j]
                    _high = high_id[j]
                    fill_mat[_low:_high+1, j] = v[2] # fill color to grids

            print(fill_mat)
            layer1 = np.zeros((self._value_window,self._time_window,), dtype=np.uint8)
            layer2 = layer1.copy()
            layer3 = layer1.copy()
            layer1 = fill_mat
            img_list = [layer2, layer1,layer3]if self._reverse_up_down_color else [layer1,layer2,layer3]
            img_mat = np.stack(img_list, axis=2)



        return img_mat
            
            