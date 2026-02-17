from dataclasses import dataclass
import yfinance as yf
import os
import pandas as pd
from pathlib import Path
from gym_trade.tool.preprocess import standardlize_df
from gym_trade.tool.get_tickers import get_tickers
from tqdm import tqdm
from queue import Queue, Empty
from threading import Event, Thread
from typing import Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError

@dataclass
class YFinanceRequest:
    symbol: str

@dataclass
class YFinanceResult:
    symbol: str
    df: pd.DataFrame

class YFinanceServer:
    def __init__(self, 
                    cache_file_pattern: Callable,
                    cache_csv_dir: Path,
                    worker_num: int = 10, 
                    interval: str = "1d", 
                    start: str | None = None, 
                    end: str | None = None,
                    period: str = 'max',
                    timeout: float = 10.0):
        self._cache_csv_dir = cache_csv_dir
        self._cache_file_pattern = cache_file_pattern
        self._worker_num = worker_num
        self._interval = interval
        self._start = start
        self._end = end
        self._period = period
        self._request_queue = Queue()
        self._result_queue = Queue()
        self._stop_event = Event()
        self._timeout = timeout
        self._workers = []
        for _ in range(self._worker_num):
            self._workers.append(Thread(target=self._worker, daemon=True))
            self._workers[-1].start()
    
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                req = self._request_queue.get(timeout=0.5)  # avoid blocking forever on close()
            except Empty:
                continue

            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(self._download, req.symbol)
                    try:
                        result = fut.result(timeout=self._timeout)
                    except TimeoutError:
                        print(f"timeout downloading {req.symbol}")
                        result = None

                self._result_queue.put(result)

            except Exception as e:
                self._result_queue.put(None)
                print(f"error downloading {req.symbol}: {e}")


    def _download(self, symbol: str) -> dict[str, pd.DataFrame]:
        assert isinstance(symbol, str), f"symbol must be a string, but got {type(symbol)}"
        df = yf.download(
                        interval = self._interval,
                        start = self._start,
                        end = self._end,
                        tickers = symbol,
                        # period = self._period,
                        multi_level_index = False,
                        threads=True, progress=False,
                        )
        if len(df.index) == 0:
            print(f"no data found for {symbol}")
            return None
        df_name = self._cache_file_pattern(symbol, self._start, self._end)
        cache_file =self._cache_csv_dir /  (df_name + ".csv")
        df.to_csv(str(cache_file))
        return YFinanceResult(symbol=symbol, df=df)
    
    def download(self, symbols: list[str]):
        pbar = tqdm(total=len(symbols), desc="downloading data")
        req_idx = 0
        dfs = {}
        while(
                not self._stop_event.is_set() 
                and pbar.n < len(symbols)
            ):
            if not self._request_queue.full() and req_idx < len(symbols):
                self._request_queue.put(YFinanceRequest(symbol=symbols[req_idx]))
                req_idx += 1
            if not self._result_queue.empty():
                try:
                    result = self._result_queue.get(
                        timeout=0.01
                    )  # 定期醒来检查 stop_event   
                    if result is not None:
                        dfs[result.symbol] = result.df
                    pbar.update(1)
                except Empty:
                    pass
                except Exception as e:
                    print(f"error getting result: {e}")
                    pass
        pbar.close()
        return dfs

    def close(self):
        self._stop_event.set()
        for w in self._workers:
            w.join()
            



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


    dfs = {}
    cache_csv_dir = Path(cache_dir) / interval 
    cache_csv_dir.mkdir(parents=True, exist_ok=True)
    if symbols is None:
        _symbols = get_tickers()
    else:
        _symbols = symbols 

    


    # _symbols = _symbols[:30]
    if data_api == 'yfinance' and interval == '1d':
        dfs = {}  
        def cache_file_pattern(symbol: str, start: str | None = None, end: str | None = None) -> str:
            s_n = start if start is not None else "smax"
            e_n = end if end is not None else "emax"
            return f"{symbol}_{s_n}_{e_n}"

        for s in _symbols:
            df_name = cache_file_pattern(s, start, end)
            cache_file =cache_csv_dir /  (df_name + ".csv")
            if  cache_file.exists(): 
                # print(f"loading {s} from {cache_file}")
                df = pd.read_csv(str(cache_file))
                df.set_index('Date', inplace=True)
                dfs[s] = df
        
        # assert False, len(dfs.keys())

        download_symbols = [s for s in _symbols if s not in dfs]
        if len(download_symbols) > 0:
            # server = YFinanceServer(
            #     cache_file_pattern=cache_file_pattern,
            #     cache_csv_dir=cache_csv_dir,
            #     worker_num=10,
            #     interval=interval,
            #     start=start,
            #     end=end,
            # )
            # dfs_new = server.download(download_symbols)
            # dfs.update(dfs_new)
            # server.close()

            download_chunk = 300
            download_symbols_chunks = []
            i = 0 
            while i < len(download_symbols):
                size = min(download_chunk, len(download_symbols) - i)
                download_symbols_chunks.append(download_symbols[i:i+size])
                i += download_chunk
            for download_symbols_chunk in tqdm(download_symbols_chunks, desc="downloading data"):
                df_out = yf.download(
                                interval = interval,
                                start = '1970-01-01' if start is None else start,
                                end = end,
                                tickers = download_symbols_chunk,
                                # period = 'max',
                                multi_level_index = False,
                                progress=False,
                                )

                symbols_out = df_out["Volume"].columns

                for s in symbols_out:
                    df = df_out.xs(s, level=1, axis=1)
                    df.dropna(inplace=True)
                    if len(df.index) == 0:
                        print(f"no data found for {s}")
                        continue
                    df_name = cache_file_pattern(s, start, end)
                    cache_file =cache_csv_dir /  (df_name + ".csv")
                    df.to_csv(str(cache_file))
                    dfs[s] =  df
        
        for s, df in dfs.items():
            df = standardlize_df(df) 
            dfs[s] = df


  

    # redownload = False
    # for symbol in tqdm(_symbols, desc="loading data"):
    #     s_n = start if start is not None else "smax"
    #     e_n = end if end is not None else "emax"
    #     df_name = f"{symbol}_{s_n}_{e_n}"
    #     cache_file =cache_csv_dir /  (df_name + ".csv")
    #     if  cache_file.exists() and not force_download: 
    #         print(f"loading {symbol} from {cache_file}")
    #         df = pd.read_csv(str(cache_file))
    #         df.set_index('Date', inplace=True)
    #         if len(df.index) == 0:
    #             print(f"re-download df {symbol}")
    #             redownload = True

    #     if not cache_file.exists() or redownload:     
    #         if data_api == 'yfinance':
    #             if interval == '1d':
    #                 # print(f"downloading {symbol} from {start} to {end}")
                    
    #                 df = yf.download(
    #                     interval = interval,
    #                     start = start,
    #                     end = end,
    #                     tickers = symbol,
    #                     period = 'max',
    #                     multi_level_index = False)
    #             else:
    #                 raise NotImplementedError
    #             if len(df.index) == 0:
    #                 print(f"no data found for {symbol} from {start} to {end}, might be proxy {proxy} is not working or start or end is not valid")
    #                 continue

    #             if cache_save:
    #                 df.to_csv(str(cache_file))
            
    #         else:
    #             raise NotImplementedError
    #     # print(df)


    #     # if cfg.mode.start is not None:
    #     #     date = datetime.strptime(cfg.mode.start, "%Y-%m-%d")
    #     #     if cfg.data.interval == "1m":
    #     #         date = date.replace(hour=9, minute=30)

    #     #     df = df.truncate(before=date)
    #     # if cfg.mode.end is not None:
    #     #     date = datetime.strptime(cfg.mode.end, "%Y-%m-%d")
    #     #     if cfg.data.interval == "1m":
    #     #         date = date.replace(hour=4, minute=00)
    #     #     df = df.truncate(after=date)
    #     # _dfs[k] = df
    #     if len(df.index) == 0:
    #         continue
    
    #     df = standardlize_df(df) 
    #     dfs[df_name] =  df
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