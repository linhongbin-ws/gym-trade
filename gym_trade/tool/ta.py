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
    return seri
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