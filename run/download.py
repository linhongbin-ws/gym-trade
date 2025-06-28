import yfinance as yf
from gym_trade.tool.get_tickers import get_tickers
from datetime import timedelta, datetime
from os.path import join
from pathlib import Path
import pandas_market_calendars as mcal
import pytz
from tqdm import tqdm
import argparse

# from stock_trader.strategy.tool.utility import load_json, PACKAGE_CONFIG_PATH, CURRENT_SYS

import os
proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy 
os.environ['HTTPS_PROXY'] = proxy 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# supress this warning:
# FutureWarning: Indexing a timezone-naive DatetimeIndex with a timezone-aware datetime is deprecated and will raise KeyError in a future version.  Use a timezone-naive object instead.
#   _d = df.truncate(before=_s, after=_e)
#


#==========================
parser = argparse.ArgumentParser(usage='Download minute data of US stock with YFinance\n')

parser.add_argument('-s', help='start date, example: 2021-01-01',required=True)
parser.add_argument('-e', help='start date, example: 2021-01-02',default=None)
parser.add_argument('-p', help='root path to save result',default="/home/ben/ssd/data/stock-data/us-minute")
args = parser.parse_args()


def request(symbol, start_date:str, end_date:str, save_root_dir:str):
    _start_date = datetime.strptime(start_date, "%Y-%m-%d")
    _end_date = datetime.strptime(end_date, "%Y-%m-%d")

    assert _start_date<=_end_date
    us_tz = pytz.timezone('US/Eastern')
    hk_tz = pytz.timezone("Hongkong")
    set_clock_hk = lambda date, hour: (us_tz.localize(date.replace(hour=hour, minute=0))).astimezone(hk_tz)
    set_clock_us = lambda date, hour: us_tz.localize(date.replace(hour=hour, minute=0))

    # start_hour = 4 
    # end_hour = 20
    start_hour = 9 
    end_hour = 16

    # print(set_clock_us(_start_date, start_hour-1))
    # print(set_clock_hk(_start_date, start_hour-1))
    df = yf.download(
        tickers = symbol,
        # start = set_clock(start, start_hour),
        # end = set_clock(end, end_hour),
        start = _start_date,
        end =_end_date, # -1 and +1 is to give buffer space
        interval = "1m",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = False,
        proxy = None,
        progress = False,
        ignore_tz=True,
        # show_errors=False
    )
    _date_idx = _start_date
    while _date_idx<=_end_date: 
        root_save_path = Path(save_root_dir) / ("kminute-"+_date_idx.strftime("%Y-%m-%d"))
        file_name = _date_idx.strftime("%Y-%m-%d") + '-' +symbol+'.csv'
        
        # _s = set_clock_us(_date_idx, start_hour)
        # _e = set_clock_us(_date_idx, end_hour)


        _d = df.truncate(before=_date_idx, after=_date_idx+timedelta(days=1))
        if _d.shape[0]>0:
            root_save_path.mkdir(parents=True, exist_ok=True)
            # print(_d)
            # print(root_save_path)
            _d.to_csv(str(root_save_path / file_name))
        _date_idx += timedelta(days=1)



start_date = args.s
end_date = args.e or datetime.today().strftime("%Y-%m-%d")
save_root_dir = args.p

cal = mcal.get_calendar('NYSE')
shedule = cal.schedule(start_date=start_date, end_date=end_date)
start_end = [(shedule.index[i*5].strftime("%Y-%m-%d"), shedule.index[i*5+4].strftime("%Y-%m-%d")) for i in range(shedule.shape[0]//5)]


print("get symbols..")
symbols = get_tickers(should_reload_data=True)

if not shedule.shape[0] % 5 == 0:
    start_end.append((shedule.index[-(shedule.shape[0] % 5)].strftime("%Y-%m-%d"), shedule.index[-1].strftime("%Y-%m-%d")))
print("===========================")
print(f"|start date: {start_date}")
print(f"|end date:   {end_date}")
print(f"|save path:  {save_root_dir}")
print(f"|symbol num: {len(symbols)}")
print("===========================")


es = {}
pbar = tqdm(start_end, position=0)
for start, end in pbar:
    pbar.set_description(f"download 5 days: {start} ~ {end}")
    for symbol in tqdm(symbols, position=1):
        try:
            request(symbol, start, end ,save_root_dir)
        except Exception as e: 
            print(e)