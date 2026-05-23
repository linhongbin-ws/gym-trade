from dataclasses import dataclass
import yfinance as yf
import os
import time
import logging
import re
from collections import defaultdict
import pandas as pd
from pathlib import Path
from gym_trade.tool.preprocess import standardlize_df
from gym_trade.tool.get_tickers import get_tickers
from tqdm import tqdm
from queue import Queue, Empty
from threading import Event, Thread
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError


_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


class _YFErrorCounter(logging.Handler):
    """Silently count yfinance errors by category instead of printing them."""

    _PATTERNS = [
        (r"possibly delisted",          "delisted"),
        (r"HTTP Error 404",             "404 not found"),
        (r"HTTP Error 4",               "4xx client error"),
        (r"HTTP Error 5",               "5xx server error"),
        (r"No data found",              "no data"),
        (r"JSONDecodeError|json",       "json error"),
        (r"ConnectionError|timed out",  "network error"),
    ]

    def __init__(self):
        super().__init__()
        self.counts: dict[str, int] = defaultdict(int)

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for pattern, label in self._PATTERNS:
            if re.search(pattern, msg, re.IGNORECASE):
                self.counts[label] += 1
                return
        self.counts["other"] += 1

    def print_summary(self) -> None:
        if not self.counts:
            return
        total = sum(self.counts.values())
        parts = ", ".join(f"{label}={n}" for label, n in sorted(self.counts.items()))
        tqdm.write(f"[yf errors] total={total}  {parts}")


def _yf_download_timed(**kwargs) -> tuple[pd.DataFrame, float]:
    """Run yf.download() and return (result, elapsed_seconds)."""
    t0 = time.time()
    df = yf.download(**kwargs)
    return df, time.time() - t0

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
            cache_dir: str | None = ".cache",
            market: str = "us-stock",
            cache_save: bool = True,
            force_download: bool | str = False,
            cache_only: bool = False,
            local_data_dir: str | None = None,
            ) -> list[pd.DataFrame]:
    if proxy is not None:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
        print(f"set proxy to {proxy}")

    dfs = {}

    if data_api == 'local' and interval == '1m':
        assert local_data_dir is not None, "local_data_dir must be set for data_api='local'"
        local_dir = Path(local_data_dir)
        assert local_dir.exists(), f"local_data_dir not found: {local_data_dir}"

        # Collect CSV files grouped by symbol.
        # Directory structure: <local_dir>/kminute-YYYY-MM-DD/YYYY-MM-DD-SYMBOL.csv
        symbol_files: dict[str, list[Path]] = {}
        for date_dir in sorted(local_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for csv_file in sorted(date_dir.glob("*.csv")):
                # stem: "YYYY-MM-DD-SYMBOL" → split on '-' first 3 parts are date
                parts = csv_file.stem.split('-', 3)
                if len(parts) < 4:
                    continue
                symbol = parts[3]
                if symbols and symbol not in symbols:
                    continue
                symbol_files.setdefault(symbol, []).append(csv_file)

        for symbol, files in tqdm(sorted(symbol_files.items()), desc="loading data"):
            day_dfs = []
            for f in files:
                df_day = pd.read_csv(str(f), index_col='Datetime')
                df_day = standardlize_df(df_day, interval='1m')
                day_dfs.append(df_day)
            if not day_dfs:
                continue
            df = pd.concat(day_dfs).sort_index()
            # Normalize index to tz-naive UTC so per-day tz inconsistencies after concat
            # (mixed tz-aware / tz-naive Timestamps demote DatetimeIndex → Index) don't break .tz access.
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            if start is not None:
                df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
            if end is not None:
                df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]
            if len(df) > 0:
                dfs[symbol] = df

        return dfs

    cache_csv_dir = Path(cache_dir) / f"{market}_{interval}_{data_api}"
    cache_csv_dir.mkdir(parents=True, exist_ok=True)
    if symbols is None:
        _symbols = get_tickers()
    else:
        _symbols = symbols 

    


    # _symbols = _symbols[:30]
    if data_api == 'yfinance' and interval == '1d':
        dfs = {}
        print(f"[cache] {cache_csv_dir.resolve()}")

        def _find_cache(symbol: str) -> Path | None:
            """Return existing cache file for symbol (pattern: SYMBOL_start_end.csv), or None."""
            hits = list(cache_csv_dir.glob(f"{symbol}_????-??-??_????-??-??.csv"))
            return hits[0] if hits else None

        def _save_cache(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
            """Standardize, write to tmp, then atomically replace final file."""
            df = standardlize_df(df)
            first = pd.Timestamp(df.index[0]).strftime('%Y-%m-%d')
            last  = pd.Timestamp(df.index[-1]).strftime('%Y-%m-%d')
            path  = cache_csv_dir / f"{symbol}_{first}_{last}.csv"
            tmp   = cache_csv_dir / f"{symbol}.tmp"
            df.to_csv(str(tmp))           # write fully before touching existing file
            old = _find_cache(symbol)
            if old and old.exists():
                old.unlink()
            tmp.rename(path)              # atomic on same filesystem
            return df, path

        dl_start_str = start if start is not None else "smax"
        dl_end_str   = end   if end   is not None else "emax"

        def _find_nodata(s: str) -> Path | None:
            hits = list(cache_csv_dir.glob(f"{s}_????-??-??_*.nodata"))
            return hits[0] if hits else None

        def _write_nodata(s: str) -> None:
            path = cache_csv_dir / f"{s}_{dl_start_str}_{dl_end_str}.nodata"
            path.touch()

        # ── force_download 预处理 ──────────────────────────────────────────────
        if force_download == "all":
            # 删除所有 csv + nodata，全量重下
            for f in cache_csv_dir.glob("*.csv"):
                f.unlink(missing_ok=True)
            for f in cache_csv_dir.glob("*.nodata"):
                f.unlink(missing_ok=True)
        elif force_download == "no_data":
            # 只删除 nodata 标记，让这些 symbol 重新进入下载队列
            for f in cache_csv_dir.glob("*.nodata"):
                f.unlink(missing_ok=True)

        for s in _symbols:
            if force_download != "no_data" and _find_nodata(s) is not None:
                continue  # 跳过已标记无数据的 symbol
            cf = _find_cache(s)
            if cf is not None and not force_download:
                df = pd.read_csv(str(cf), index_col=0)
                df.index = pd.to_datetime(df.index)
                dfs[s] = df

        # download all symbols not already cached and not marked nodata
        download_symbols = [
            s for s in _symbols
            if s not in dfs and _find_nodata(s) is None
        ]
        if cache_only and download_symbols:
            print(f"[cache_only] skipping download for {len(download_symbols)} symbols not in cache")
            download_symbols = []
        if len(download_symbols) > 0:
            download_chunk = 50
            download_symbols_chunks = []
            i = 0
            while i < len(download_symbols):
                size = min(download_chunk, len(download_symbols) - i)
                download_symbols_chunks.append(download_symbols[i:i+size])
                i += download_chunk
            pbar = tqdm(total=len(download_symbols), desc="downloading", unit="sym", dynamic_ncols=True)
            dl_ok = 0  # symbols with data from this download run

            yf_logger = logging.getLogger("yfinance")
            _err_counter = _YFErrorCounter()
            yf_logger.addHandler(_err_counter)
            _orig_propagate = yf_logger.propagate
            yf_logger.propagate = False

            def _save_batch(df_out, chunk):
                """Process one batch result and save symbols with data immediately."""
                nonlocal dl_ok
                if df_out is None or df_out.empty:
                    return
                if len(chunk) == 1:
                    s = chunk[0]
                    df = df_out.xs(s, level=1, axis=1) if isinstance(df_out.columns, pd.MultiIndex) else df_out
                    df = df.dropna(subset=_OHLCV_COLS, how='all')
                    if len(df.index) > 0:
                        df, _ = _save_cache(s, df)
                        dfs[s] = df
                        dl_ok += 1
                else:
                    if not isinstance(df_out.columns, pd.MultiIndex):
                        return
                    for s in df_out.columns.get_level_values(1).unique():
                        try:
                            df = df_out.xs(s, level=1, axis=1)
                        except KeyError:
                            continue
                        df = df.dropna(subset=_OHLCV_COLS, how='all')
                        if len(df.index) > 0:
                            df, _ = _save_cache(s, df)
                            dfs[s] = df
                            dl_ok += 1

            for download_symbols_chunk in download_symbols_chunks:
                df_out = None
                interrupted = False
                try:
                    df_out, _ = _yf_download_timed(
                        interval=interval,
                        period='max',
                        tickers=download_symbols_chunk,
                        threads=True,
                        progress=False,
                    )
                except KeyboardInterrupt:
                    interrupted = True

                _save_batch(df_out, download_symbols_chunk)
                pbar.update(len(download_symbols_chunk))
                pbar.set_postfix(ok=f"{dl_ok}/{len(download_symbols)}")

                if interrupted:
                    pbar.close()
                    raise KeyboardInterrupt

            # ── retry with fallback params (each phase runs once) ────────────────
            # Phase 1: start=1991-01-01, chunk=50 — widen date range
            # Phase 2: period='max',     chunk=50 — max range
            RETRY_PHASES = [
                {"label": "1991-start",  "repeat": 1, "chunk": 50, "backoff": 1.0,
                 "kwargs": {"start": "1991-01-01", "end": end}},
                {"label": "screener-range", "repeat": 1, "chunk": 50, "backoff": 3.0,
                 "kwargs": {"start": '1970-01-01' if start is None else start, "end": end}},
            ]

            def _process_chunk(df_out, chunk, still_failed):
                """Parse df_out, save hits to cache/dfs, append misses to still_failed."""
                nonlocal dl_ok
                if df_out is None or df_out.empty:
                    still_failed.extend(chunk)
                    return
                if len(chunk) == 1:
                    s = chunk[0]
                    df = (df_out.xs(s, level=1, axis=1)
                          if isinstance(df_out.columns, pd.MultiIndex) else df_out)
                    df = df.dropna(subset=_OHLCV_COLS, how='all')
                    if len(df.index) > 0:
                        df, _ = _save_cache(s, df)
                        dfs[s] = df
                        dl_ok += 1
                    else:
                        still_failed.append(s)
                else:
                    if not isinstance(df_out.columns, pd.MultiIndex):
                        still_failed.extend(chunk)
                        return
                    returned = set(df_out.columns.get_level_values(1).unique())
                    for s in chunk:
                        if s not in returned:
                            still_failed.append(s)
                            continue
                        try:
                            df = df_out.xs(s, level=1, axis=1).dropna(subset=_OHLCV_COLS, how='all')
                        except KeyError:
                            still_failed.append(s)
                            continue
                        if len(df.index) == 0:
                            still_failed.append(s)
                            continue
                        df, _ = _save_cache(s, df)
                        dfs[s] = df
                        dl_ok += 1

            retry_symbols = [s for s in download_symbols if s not in dfs]
            total_attempts = sum(p["repeat"] for p in RETRY_PHASES)
            attempt = 0
            for phase in RETRY_PHASES:
                for rep in range(phase["repeat"]):
                    if not retry_symbols:
                        break
                    attempt += 1
                    label = phase["label"]
                    chunk_size = phase["chunk"]
                    backoff = phase["backoff"] * (2 ** rep)   # 指数退避
                    time.sleep(backoff)
                    still_failed: list[str] = []
                    retry_chunks = [retry_symbols[i:i+chunk_size]
                                    for i in range(0, len(retry_symbols), chunk_size)]
                    pbar.reset(total=len(retry_symbols))
                    pbar.set_description(f"retry {attempt}/{total_attempts} [{label}]")
                    for chunk in retry_chunks:
                        try:
                            df_out, _ = _yf_download_timed(
                                interval=interval,
                                tickers=chunk,
                                threads=True,
                                progress=False,
                                **phase["kwargs"],
                            )
                        except Exception:
                            still_failed.extend(chunk)
                            pbar.update(len(chunk))
                            continue
                        _process_chunk(df_out, chunk, still_failed)
                        pbar.update(len(chunk))
                        pbar.set_postfix(ok=f"{dl_ok}/{len(download_symbols)}")
                    retry_symbols = still_failed
                if not retry_symbols:
                    break

            pbar.close()
            yf_logger.removeHandler(_err_counter)
            yf_logger.propagate = _orig_propagate
            _err_counter.print_summary()

            if retry_symbols:
                for s in retry_symbols:
                    _write_nodata(s)
                tqdm.write(f"[download] {len(retry_symbols)} symbols marked as nodata ({dl_start_str}~{dl_end_str})")

        # standardize symbols loaded from cache (newly downloaded are already standardized)
        for s in list(dfs.keys()):
            if s not in download_symbols:
                dfs[s] = standardlize_df(dfs[s])


  

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