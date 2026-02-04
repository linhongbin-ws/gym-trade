import pandas as pd
from os.path import join, isdir
from datetime import timedelta, datetime
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")
# from gym_daytrade.tool.visualizer import plot_visualize_ohlc, plot_visualize_ohlc_std_feature
from os import walk, makedirs
from tqdm import tqdm
import numpy as np
# def extract_feature_name(search_str:str, pattern_str:str)->Optional[int]:
#     start_idx = search_str.find(pattern_str)
#     if start_idx==-1:
#         return None
#     else:
#         param = int(search_str[start_idx + len(pattern_str) +1:])
#         return param


def create_feature_df(df: pd.DataFrame, feature_names: List)->pd.DataFrame:
    """ create feature signals given a ohlc dataframe

    input:
        df: dataframe with columns of ohlc, volume ,and with index of Datetime
            example:
                [Datetime         open high low close volume]
                [datetime object   x     x    x   x     x]

        feature_names: list of string for names of new features

    output:
        df: dataframe with columns of ohlc, volume, and new features, and with index of Datetime
            example:
                [Datetime       open high low close volume, close_5m_ma]
                [datetime object   x     x    x   x     x         x]
    """
    df_feature = df
    for feature_name in feature_names:
        start_idx = feature_name.find("_")
        pre_string = feature_name[:start_idx]
        post_string = feature_name[start_idx+1:]
        if pre_string == 'close-ma':
            param = int(post_string)
            if param <=0:
                raise Exception(f"Param should be larger than zero in {feature_name}")
            df_feature[feature_name] = df['close'].rolling(window=param).mean()
            continue

        elif pre_string == 'volume-ma':
            param = int(post_string)
            if param <=0:
                raise Exception(f"Param should be larger than zero in {feature_name}")
            df_feature[feature_name] = df['volume'].rolling(window=param).mean()
            continue

        elif pre_string == 'close-rel-t0-ma':
            param = int(post_string)
            if param <=0:
                raise Exception(f"Param should be larger than zero in {feature_name}")
            df_feature[feature_name] = (df['close'].rolling(window=param).mean() - df['close'].iloc[0])/df['close'].iloc[0]
            continue

        else:
            raise Exception(f"{feature_name} is not support, please add feature in source code")


        # elif feature_name == 'close_30m_ma':
        #     df_feature['close_30m_ma'] = df['close'].rolling(window=90).mean()
        # elif feature_name == 'close_3m_ma':
        #     df_feature['close_3m_ma'] = df['close'].rolling(window=3).mean()
        # elif feature_name == 'volume_3m_ma':
        #     df_feature['volume_3m_ma'] = df['volume'].rolling(window=3).mean()
        # elif feature_name == 'close_ratio':
        #     df_feature['close_ratio'] = df['close']/df['close'].shift(1)-1
        # elif feature_name == 'close_ratio_3m_ma':
        #     df_feature['close_ratio_3m_ma'] = (df['close']/df['close'].shift(1)-1).rolling(window=3).mean()
        # elif feature_name == 'close_ratio_10m_ma':
        #     df_feature['close_ratio_10m_ma'] = (df['close']/df['close'].shift(1)-1).rolling(window=10).mean()
        # elif feature_name == 'close_ratio_30m_ma':
        #     df_feature['close_ratio_30m_ma'] = (df['close']/df['close'].shift(1)-1).rolling(window=30).mean()
        # elif feature_name == 'volume_ratio':
        #     df_feature['volume_ratio'] = df['close']/df['close'].shift(1)-1
        # elif feature_name == 'volume_ratio_10m_ma':
        #     df_feature['volume_ratio_10m_ma'] = (df['volume']/df['volume'].shift(1)-1).rolling(window=10).mean()
        # elif feature_name == 'hl_win_ratio':
        #     df_feature['hl_win_ratio'] = df_feature['close']
        #     for i in range(len(df_feature.index)):
        #         high = df_feature['high'].iloc[:i+1].max()
        #         low = df_feature['low'].iloc[:i + 1].min()
        #         df_feature['hl_win_ratio'].iloc[i] = ((df_feature['close'].iloc[i]-low)/(high-low) - 0.5)*2
        # elif feature_name == 'flow_direction':
        #     _df = df['close']-df['close'].shift(1)
        #     _df[_df>0] = 1
        #     _df[_df < 0] = -1
        #     df['flow_direction'] = _df.rolling(window=6).mean()
        # elif feature_name == 'expected_value_6m':
        #     discount_value = 1
        #     horizon = 15
        #     tmp = None
        #     _df = None
        #
        #     for i in range(1,horizon):
        #         _tmp = (df_feature['close'].shift(-i)-df_feature['close'])/df_feature['close']*pow(discount_value, i)
        #         _df = _tmp if _df is None else pd.concat([_df, _tmp], axis=1)
        #
        #     df_feature['expected_value_6m'] = _df.mean(axis=1)
        #     df_feature['expected_value_6m_std'] = _df.std(axis=1)
        # else:
        #     print("the feature name {} is not available, please check avaliable features!".format(feature_name))

    return df_feature

def standardlize_df(df: pd.DataFrame, interval: str = "1d" )-> pd.DataFrame:
    """ standardlize dataframe

    example:
                                open   high   low   close  volume
        datetime                                                     
        2021-01-04 09:30:00-05:00  5.610  5.610  5.61  5.6100       0
        ...                          ...    ...   ...     ...     ...
        2021-01-04 15:59:00-05:00  5.590  5.600  5.58  5.6000   10636
    
    """
    _df = df.copy()
    _df.columns= _df.columns.str.lower() # all index names to lower case
    if _df.index.name is not None:
        _df.index.name = _df.index.name.lower() 

    _df = _df[['open', 'high', 'low', 'close', 'volume']]
    # for col in _df.columns: 
    #     assert col in ['date', 'open', 'high', 'low', 'close', 'volume'], f"column {col} is not supported" 
    # change names

    # if 'date' in _df.columns.values:
    #     _df.set_index('date', inplace=True)

    # to datetime
    assert len(_df.index) !=0    
    _df.index = pd.to_datetime(_df.index)
    # print(type(_df.index.values[0]))
    assert isinstance(_df.index.values[0], (np.datetime64, datetime)), type(_df.index.values[0])
    _df.sort_index(inplace=True)

    if interval == "1m":
        _df = fill_missing_frame(_df) 
        

    return _df

def fill_missing_frame(df: pd.DataFrame)-> pd.DataFrame:
    """ filling missing row according to datetime order with last frame close value and zero volume

    input:
        df: dataframe with columns of ohlc, volume
            example:
                [Datetime               open high low close volume]
                [2020-12-01 12:01:00   o     h    l   c     v]
                [2020-12-01 12:04:00   x     x    x   x     x]
                [2020-12-01 12:05:00   x     x    x   x     x]

        feature_names: list of string for names of new features

    output:
        df: dataframe with columns of ohlc, volume, and new features, and with index of Datetime
            example:
                [Datetime               open high low close volume]
                [2020-12-01 12:01:00   o     h    l   c     v]
                [2020-12-01 12:02:00   c     c    c   c     0]
                [2020-12-01 12:03:00   c     c    c   c     0]
                [2020-12-01 12:04:00   x     x    x   x     x]
                [2020-12-01 12:05:00   x     x    x   x     x]
    """
    start = df.index[0].replace(hour=9, minute=30)
    end = df.index[0].replace(hour=15, minute=59)
    idx = pd.date_range(start, end, freq='min')
    df_new = df.reindex(idx, fill_value=np.nan)
    df_new.index.name = df.index.name
    df_new['is_nan'] = df_new.isna().any(axis=1)
    # print(df)
    if df_new['is_nan'].any():
        critical_nan = df_new[df_new['is_nan']&(~df_new['is_nan']).shift(1)] # the beginning indexes if there is a continous nans
        critical_nonan = df_new.loc[critical_nan.index - timedelta(minutes=1)]
        df_new['open'].loc[critical_nan.index] = critical_nonan['close'].values
        df_new['high'].loc[critical_nan.index] = critical_nonan['close'].values
        df_new['low'].loc[critical_nan.index] = critical_nonan['close'].values
        df_new['close'].loc[critical_nan.index] = critical_nonan['close'].values
        df_new['volume'].loc[critical_nan.index] = 0

        # if df_new['is_nan'].iloc[0]:
        #     critical_nan = df_new[df_new['is_nan'] & (~df_new['is_nan']).shift(-1)]
        #     critical_nonan = df_new.loc[critical_nan.index + timedelta(minutes=1)]
        #     idx_nona = critical_nonan.index[0]
        #     df_new['open'].iloc[0] =  df_new['close'].loc[idx_nona]
        #     df_new['high'].iloc[0] = df_new['close'].loc[idx_nona]
        #     df_new['low'].iloc[0] = df_new['close'].loc[idx_nona]
        #     df_new['close'].iloc[0] = df_new['close'].loc[idx_nona]
        #     df_new['volume'].iloc[0] = 0

        df_new.fillna(method='ffill',inplace=True)
        # df_new.fillna(method='bfill',inplace=True)

    df_new.sort_index(inplace=True)
    df_new.drop(columns='is_nan', inplace=True)
    return df_new


def data_preprocess(from_csv_file:str, add_feature_names:List[str], is_plot=False)->pd.DataFrame:
    """Pipline of data process, a top-level wrapper function

    input:
        :param from_csv_file: the file path to the  csv file that needs to be processed
        :param add_feature_names:  create features signals
        :param is_plot (optional): True if plot

    output
        :return: dataframe after data processing
            example:
                [Datetime       open high low close volume, close_5m_ma]
                [datetime object   x     x    x   x     x         x]
    """
    df = pd.read_csv(from_csv_file)
    df = df.rename(columns={'Datetime': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    ## fill missing minute data frame with previous frame
    df = fill_missing_frame(df)
    # plot_visualize_ohlc(df)

    ## create new feature
    df_features = create_feature_df(df, add_feature_names)
    # if is_plot:
    #     plot_visualize_ohlc(df_features, add_feature_names)

    df_features.index.name = "date"
    return df_features

def data_preprocess_folder(from_dir:str, to_dir:str, feature_names:List[str], isTry=True)->None:
    """ data process all the file in 'from_dir' directory and save result csv to 'to_dir'

    input:
        :param from_dir: the directory to all the files that needs to be data processed
        :param to_dir:  the directory for saving result csv
    """
    for root, dirs, files in walk(from_dir, topdown=False):
        for file in tqdm(files):
            if file.endswith(".csv"):
                file_name = join(root, file)
                if isTry:
                    try:
                        df_process = data_preprocess(file_name, feature_names, is_plot=None)
                        if not isdir(to_dir):
                            makedirs(to_dir)
                        df_process.to_csv(join(to_dir, file))

                    except Exception as e: print(e)
                else:
                    df_process = data_preprocess(file_name, feature_names,
                                                 is_plot=None)
                    if not isdir(to_dir):
                        makedirs(to_dir)
                    df_process.to_csv(join(to_dir, file))


if __name__ == "__main__":
    # test case
    is_data_process_func = True
    is_test_data_process_folder_func = False
    is_plot_features = False

    if is_data_process_func:
        # settings
        file_name = r'E:\stock-data\us-miniute\kminute-2021-05-12\2021-05-12-DEH.csv'
        feature_names = ['close-ma_3','close-ma_5']

        df_features = data_preprocess(file_name, feature_names, is_plot=True)
        print(df_features)

    if is_test_data_process_folder_func:
        feature_names = ['close_3m_ma', 'close_5m_ma','close_ratio','close_ratio_3m_ma','volume_ratio_10m_ma','hl_win_ratio']
        from_dir = join("E:", "stock-data","us-miniute","kminute-2021-01-22")
        to_dir = join("E:", "stock-data","us-miniute","test","data_process")
        data_preprocess_folder(from_dir, to_dir, feature_names)

    if is_plot_features:
        # settings
        file_name = join("E:", "stock-data","us-miniute","kminute-2021-01-22",'2021-01-22-GME.csv')
        feature_names = ['close_3m_ma', 'close_5m_ma','close_ratio','close_ratio_3m_ma','volume_ratio_10m_ma','hl_win_ratio','expected_value_6m','close_ratio_30m_ma','flow_direction']
        df_features = data_preprocess(file_name, feature_names, is_plot=False)
        # feature_names.append('expected_value_6m_std')
        df_features['expected_value_6m'] =  df_features['expected_value_6m'].shift(15)
        df_features['expected_value_6m_std'] = df_features['expected_value_6m_std'].shift(15)
        df_features['close_line'] = df_features['close']
        feature_names.append('close_line')
        # plot_visualize_ohlc_std_feature(df_features,feature_names)
        print(df_features)