import pandas as pd
from os.path import isfile
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
import pandas_ta as ta
import numpy as np


def ma(df: pd.DataFrame, key:str,window:int)->pd.Series:
    """ moving average """
    _df = df.copy()
    assert window>0 and key in df.columns
    # feature_name = 'ma_'+key+'_'+str(window)
    seri = _df[key].rolling(window=window).mean()
    return {'value': seri}

def break_high(df: pd.DataFrame, key, ):
    _df = df.copy()
    _df['new_high'] = np.nan
    _new_high = None
    for ind in _df.index:
        if _new_high is None:
            _new_high = _df[key][ind]
            value = 1
        elif _new_high <  _df[key][ind]:
            _new_high = _df[key][ind]
            value = 1
        else:
            value = 0
            
        _df['new_high'][ind] = value
    return _df['new_high']

def rolling_op(df, key, operation, window):
    _df = df.copy()
    if operation == "or":
        seri = _df[key].rolling(window=window).max().astype(bool) # equivalent to or operation
    else:
        raise NotImplementedError
    return seri

def direction(df, key):
    _df = df.copy()
    seri = (_df[key] - _df[key].shift(1))>0
    return seri

def trade_curb(df):
    _df = df.copy()
    seri1 = _df['high'] == _df['low']
    seri2 = _df['volume'] <=0
    seri = seri1 & seri2
    return seri



def direction_toggle(df, key):
    # direction of a key is change and it will give True, otherwise False
    _df = df.copy()
    upsign = _df[key] > _df[key].shift(1)
    change_sign = upsign ^ upsign.shift(1)
    change_sign_shift = change_sign.shift(-1)
    change_sign_shift.fillna(False)
    change_sign_shift.iloc[-1] = True

    return_list = {}
    return_list['direction_toggle_bool@'+key] =  np.int32(change_sign_shift) # 1 or 0
    return_list['direction_toggle_value@'+key] =  _df[key].copy()
    return_list['direction_toggle_value@'+key][np.logical_not(change_sign_shift)] = np.nan

    # pattern id
    # 0 : index <3

    # pattern 1
    #       ^
    #   ^  
    #     v
    # v 

    # pattern 2
    #       ^
    #   ^  
    # v    
    #     v

    # pattern 3
    #   ^    
    #       ^
    #     v 
    # v    

    # pattern 4
    #   ^    
    #       ^
    # v     
    #     v

    # pattern 5
    #   ^    
    # v      
    #       ^
    #     v      

    # pattern 6
    # ^      
    #     ^  
    #   v     
    #       v 

    # pattern 7
    #     ^   
    # ^      
    #   v     
    #       v
 

    # pattern 8
    # ^      
    #     ^   
    #       v  
    #   v    

    # pattern 9
    #     ^   
    # ^      
    #       v  
    #   v   

    # pattern 10
    #     ^   
    #       v 
    # ^        
    #   v    
    pattern_sr = return_list['direction_toggle_value@'+key].copy()
    pattern_sr[:] = np.nan
    for pt_idx, _ in enumerate(pattern_sr):
        new_seri = return_list['direction_toggle_value@'+key][:pt_idx+1].dropna().copy()
        if len(new_seri.index) < 4:
            pt_value = 0
        else:
            if np.isnan(return_list['direction_toggle_value@'+key][pt_idx]):
                pivot = new_seri.iloc[-3:].tolist()
            else:
                pivot = new_seri.iloc[-4:-1].tolist()
            pivot.append(_df[key][pt_idx])
            if pivot[3]>pivot[2]:
                if pivot[3]>pivot[1] and pivot[2]>pivot[0]:
                    # date = pattern_sr.index[pt_idx]
                    pt_value = 1
                elif pivot[3]>pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 2
                elif pivot[3]<pivot[1] and pivot[2]>pivot[0]:
                    pt_value = 3
                elif pivot[3]<pivot[1] and pivot[2]<pivot[0] and pivot[3]>pivot[0]:
                    pt_value = 4
                else:
                    pt_value = 5
            else:
                if pivot[3]<pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 6
                elif pivot[3]<pivot[1] and pivot[2]>pivot[0]:
                    pt_value = 7
                elif pivot[3]>pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 8
                elif pivot[3]>pivot[1] and pivot[2]>pivot[0] and pivot[3]<pivot[0]:
                    pt_value = 9
                else:
                    pt_value = 10
        pattern_sr.iloc[pt_idx] = pt_value
    
    return_list['direction_toggle_pattern_id@'+key] = pattern_sr
    return_list['direction_toggle_pattern_strongup@'+key] = np.int32(pattern_sr ==1)

    strong_up_acc = return_list['direction_toggle_value@'+key].copy()
    strong_up_acc[:] = np.nan
    cnt = 0
    for idx, _ in enumerate(strong_up_acc):
        if idx == 0:
            strong_up_acc[idx] = 0
        else:
            if pattern_sr[idx] in [1] and  pattern_sr[idx] !=  pattern_sr[idx-1]:
                cnt +=1
            elif pattern_sr[idx] in [6,7,8]:
                cnt=0
            strong_up_acc[idx] = cnt
    return_list['direction_toggle_pattern_strongup_acc@'+key] = strong_up_acc

    return return_list