"""
Find stock in play.
This script will return stock-in-play symbols with dates.
"""

import sys
from os.path import join
import pandas as pd
import argparse
from tqdm import tqdm
from copy import deepcopy
from gym_trade.tool import screen

# parser = argparse.ArgumentParser(usage='Find stocks in play using daily K data of US stocks\n',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-p','--path', help='path to load hdf file', required=True)
# parser.add_argument('-s','--start', help='start time', default=None)
# parser.add_argument('-e','--end', help='end time', default=None)
# args = parser.parse_args()

path = "~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5"

def find_stock_in_play(df_meta:pd.DataFrame):

    symbols = df_meta.columns.levels[0][:20]
    pbar = tqdm(symbols)
    # fil_func_strs = {'pre_gap': {'ratio_lower_bd':0.02}}
    results = {}
    for symbol in pbar:
        df = deepcopy(df_meta[symbol])
        df.dropna(inplace=True)

        screen_func = getattr(screen, "new_high")
        screen_args = {"df": df, "period": 52}
        df = screen_func(**screen_args)

        if df is not None:
            results[symbol] = pd.to_datetime(df.index)

    print(results)
    return results

df_meta = pd.read_hdf(path)
results = find_stock_in_play(df_meta)



