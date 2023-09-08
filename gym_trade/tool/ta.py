import pandas as pd
from os.path import isfile
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
import pandas_ta as ta
import numpy as np


def ma(df: pd.DataFrame, input_column_name:str,window:int)->pd.DataFrame:
    """ moving average """
    _df = df.copy()
    assert window>0 and input_column_name in df.columns
    feature_name = 'ma_'+input_column_name+'_'+str(window)
    _df[feature_name] = _df[input_column_name].rolling(window=window).mean()
    return _df, [feature_name]

def chg(df: pd.DataFrame, input_column_name:str):
    """ perentage of change """
    _df = df.copy()
    assert input_column_name in df.columns

    feature_name = 'chg_'+input_column_name
    _df[feature_name] =_df[input_column_name].pct_change()
    return _df, [feature_name]

def rsi(df: pd.DataFrame, input_column_name:str, length:int):
    _df = df.copy()
    assert length>0 and input_column_name in df.columns
    feature_name = 'rsi_'+input_column_name+'_'+str(length)
    _df[feature_name] = ta.rsi(_df[input_column_name], length=length)
    return _df, [feature_name]

def true_chg_rate(df: pd.DataFrame, short_range: int, long_range:int):
    _df = df.copy()
    _df['close_true_chg_abs'] =  (_df['close'] -  _df['close'].shift(1)).abs()
    _df['close_true_chg_abs'].iloc[0] = np.abs(_df['close'].iloc[0] - _df['open'].iloc[0])
    # print(_df)
    feature_name = 'true_chg_rate'
    _df[feature_name] = np.nan
    for index in range(len(_df.index)):
        if index < short_range -1:
            continue
        _range = np.clip(index+1,None,long_range)
        _seri = _df['close_true_chg_abs'].iloc[index+1-_range:index+1].copy()
        _val = (_seri[-1] - _seri.mean()) / _seri.std()
        _df[feature_name].iloc[index] = _val
    return _df, [feature_name]

def true_chg_ma(df: pd.DataFrame, short_length:int, long_length: int):
    _df = df.copy()
    _df['close_true_chg_abs'] =  (_df['close'] -  _df['close'].shift(1)).abs()
    _df['close_true_chg_abs'].iloc[0] = np.abs(_df['close'].iloc[0] - _df['open'].iloc[0])
    feature_name = 'true_chg_ma'
    _df[feature_name] =  _df['close_true_chg_abs'].rolling(long_length).mean()

    for i in range(short_length, long_length):
        _df[feature_name].iloc[i-1] = _df['close_true_chg_abs'].iloc[:i].rolling(i).mean().iloc[-1]


    return _df, [feature_name] 

def donchain(df: pd.DataFrame,
              short_length,
              long_length):
    _df = df.copy()
    donchaindf = _df.ta.donchian(lower_length=long_length, 
                                upper_length=long_length)
    # print(donchaindf)
    for i in range(short_length, long_length):
        _dc_df = _df.iloc[:i].ta.donchian(lower_length=i, 
                                upper_length=i).dropna()

        _dc_df.columns = donchaindf.columns.tolist()
        for index, row in _dc_df.iterrows():
            donchaindf.loc[index] = row

    # print(donchaindf)
    for column in donchaindf:
        _df[column] = donchaindf[column]
    feature_names = [column for column in donchaindf]
    return _df, feature_names

def rvi(df: pd.DataFrame,
        length:int):
    _df = df.copy()
    feature_name = 'rvi'+'_'+str(length)
    _df[feature_name] = _df.ta.rvi(length=length)
    return _df, [feature_name]

def thermo(df: pd.DataFrame, **kwargs):
    _df = df.copy()
    # feature_name = 'rvi'+'_'+str(length)
    print(_df.ta.thermo(**kwargs))
    # return _df, [feature_name]
def massi(df: pd.DataFrame, **kwargs):
    _df = df.copy()
    post_name = ['_'+str(v)  for k,v in kwargs.items()]
    feature_name = 'massi'
    if len(post_name) > 0:
        feature_name = feature_name + post_name
    # print(_df.ta.massi(**kwargs))
    _df[feature_name] = _df.ta.rvi(**kwargs)
    return _df, [feature_name]

def high_close(df: pd.DataFrame, **kwargs):
    _df = df.copy()
    feature_name = 'high_close'
    _df[feature_name] = np.nan
    for index in range(len(_df.index)):
        _seri = _df[['close', 'open']].iloc[:index+1]
        _val = _seri.max().max()
        # print(_val)
        _df[feature_name].iloc[index] = _val
    return _df, [feature_name]

def atr(df:pd.DataFrame, **kwargs):
    _df = df.copy()
    feature_name = "atr"
    _df["atr"] = _df.ta.atr(**kwargs)
    return _df, [feature_name] 

def add(df:pd.DataFrame, feature1:str, feature2:str, scale1:float, scale2:float, **kwargs):
    _df = df.copy()
    feature_name = feature1 + '_add_' + feature2 + '_'+ str(scale1) + '_' + str(scale2)
    _df[feature_name] = scale1 * _df[feature1]  + scale2 * _df[feature2]
    return _df, [feature_name] 

if __name__ == '__main__':
    # csv_file = '/home/ben/ssd/data/stock-data/us-minute/kminute-2021-01-04/2021-01-04-OCSL.csv'
    csv_file = './data/screen-gap/2020-12-29-IRIX.csv'
    assert isfile(csv_file)
    df = pd.read_csv(csv_file)
    print(df)
    df = standardlize_df(df)
    print(df)
    df = fill_missing_frame(df)
    print(df)
    df, feature_names  = ma(df, 'close',10)
    print(df)
    print(feature_names)
    df, feature_names = chg(df, 'close')
    print(df)
    print(feature_names)
    df, feature_names = rsi(df, 'close', 15)
    print(df)
    print(feature_names)

    # df, feature_names = true_chg_rate(df, 3, 60)
    # print(df)
    # print(feature_names)
    # # print(df.ta.indicators())

    # df, feature_names = donchain(df, 2, 12)
    # print(df)
    # print(feature_names)
    
    df, feature_names = rvi(df, length=15)
    print(df)
    print(feature_names)

    thermo(df, length=15)
    df, feature_names = massi(df)
    print(df)
    print(feature_names)


    df, feature_names = high_close(df)
    print(df)
    print(feature_names)

    df, feature_names = atr(df)
    print(df)
    print(feature_names)

    df, feature_names = add(df, 'high_close', 'atr', 1, -3)
    print(df)
    print(feature_names)

    df, feature_names = true_chg_ma(df, 1, 10)
    print(df)
    print(feature_names)