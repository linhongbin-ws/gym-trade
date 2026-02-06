import yfinance as yf
import os
import pandas as pd
from pathlib import Path
from gym_trade.tool.preprocess import standardlize_df

def load_data(
            data_api: str = "yfinance",
            proxy: str | None = "http://127.0.0.1:7897",
            interval: str = "1d",
            start: str | None = None,
            end: str | None = None,
            symbols: list[str] = [],
            cache_dir: str | None = "/tmp/gym_trade",
            cache_save: bool = True,
            force_download: bool = False,
            local_data_dir: str | None = None,
            ) -> list[pd.DataFrame]:
    if proxy is not None:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
        print(f"set proxy to {proxy}")


    dfs = []
    cache_csv_dir = Path(cache_dir) / interval 
    cache_csv_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        s_n = start if start is not None else "smax"
        e_n = end if end is not None else "emax"
        cache_name = f"{symbol}_{s_n}_{e_n}.csv"
        cache_file =cache_csv_dir /  cache_name
        if  cache_file.exists() and not force_download: 
            print(f"loading {symbol} from {cache_file}")
            df = pd.read_csv(str(cache_file))
            df.set_index('Date', inplace=True)
        elif data_api == 'yfinance':
            if interval == '1d':
                print(f"downloading {symbol} from {start} to {end}")
                
                df = yf.download(
                    interval = interval,
                    start = start,
                    end = end,
                    tickers = symbol,
                    multi_level_index = False)
            else:
                raise NotImplementedError


            assert len(df.index) > 0, f"no data found for {symbol} from {start} to {end}, might be proxy {proxy} is not working"
            if cache_save:
                df.to_csv(str(cache_file))
        
        elif data_api == 'local_1m':
            pass
        else:
            raise NotImplementedError
        # print(df)
        df = standardlize_df(df) 
        dfs.append(df)
    return dfs

if __name__ == "__main__":
    dfs = load_data(
                data_api="yfinance", 
            proxy="http://127.0.0.1:7897", interval="1d", 
            start="2021-01-02", 
            end=None, symbols=["AAPL"])
    print("finish 1")
    dfs = load_data(
                data_api="yfinance", 
            proxy="http://127.0.0.1:7897", interval="1d", 
            start="2021-01-02", 
            end=None, symbols=["AAPL"], force_download=True)
    print("finish 2")
    dfs = load_data(
                data_api="yfinance", 
            proxy="http://127.0.0.1:7897", interval="1d", 
            start="2021-01-02", 
            end=None, symbols=["AAPL"])
    print("finish 3")
    print(dfs)