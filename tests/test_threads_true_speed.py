"""
测试 yf.download(threads=True) 下载全量 symbol 的速度
- NASDAQ 全量 symbol
- daily interval，2020-01-01 到今天
- 300 个一批
- 记录每批耗时、成功率、总耗时
"""

import os, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import date
from pathlib import Path
from collections import defaultdict

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── proxy ────────────────────────────────────────────────────────────────────────
PROXY = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"]  = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# ── parameters ──────────────────────────────────────────────────────────────────
START         = "2020-01-01"
END           = date.today().isoformat()
BATCH_SIZE    = 300
BATCH_TIMEOUT = 120
BATCH_SLEEP   = 2.0
REPORT_PATH   = Path("report_threads_true_speed.txt")

# ── symbol list ──────────────────────────────────────────────────────────────────
print("\nFetching symbol list …")
try:
    from gym_trade.tool.get_tickers import get_tickers
    ALL_SYMBOLS = list(dict.fromkeys(get_tickers(should_reload_data=True)))
    print(f"  NASDAQ API → {len(ALL_SYMBOLS)} symbols")
except Exception as e:
    print(f"  NASDAQ API failed ({e}), using fallback list")
    ALL_SYMBOLS = [
        "AAPL","MSFT","AMZN","GOOG","META","NVDA","TSLA","BRK-B","JPM","V",
        "UNH","XOM","JNJ","WMT","MA","PG","HD","CVX","MRK","ABBV",
    ]

BATCHES = [ALL_SYMBOLS[i:i+BATCH_SIZE] for i in range(0, len(ALL_SYMBOLS), BATCH_SIZE)]
print(f"  {len(ALL_SYMBOLS)} symbols  →  {len(BATCHES)} batches (size={BATCH_SIZE})\n")

# ── download one batch ────────────────────────────────────────────────────────────
def download_batch(symbols: list[str]) -> dict[str, str]:
    """Returns {symbol: 'ok' | failure_reason}"""
    result: dict[str, str] = {}

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                yf.download,
                symbols,
                start=START, end=END,
                interval="1d",
                threads=True,
                progress=False,
                multi_level_index=True,
            )
            try:
                df_out = fut.result(timeout=BATCH_TIMEOUT)
            except FuturesTimeout:
                for s in symbols:
                    result[s] = "TIMEOUT"
                return result
    except Exception as e:
        err = str(e)
        tag = "RATE_LIMITED" if ("RateLimit" in err or "Too Many" in err) else f"NETWORK_ERROR"
        for s in symbols:
            result[s] = tag
        return result

    if df_out is None or df_out.empty:
        for s in symbols:
            result[s] = "EMPTY_RESPONSE"
        return result

    if len(symbols) == 1:
        s = symbols[0]
        if isinstance(df_out.columns, pd.MultiIndex):
            try:
                df = df_out.xs(s, level=1, axis=1)
                result[s] = "ok" if not df.dropna(how="all").empty else "ALL_NaN"
            except KeyError:
                result[s] = "NOT_IN_OUTPUT"
        else:
            result[s] = "ok" if not df_out.dropna(how="all").empty else "ALL_NaN"
        return result

    if not isinstance(df_out.columns, pd.MultiIndex):
        for s in symbols:
            result[s] = "EMPTY_RESPONSE"
        return result

    returned = set(df_out.columns.get_level_values(1).unique())
    for s in symbols:
        if s not in returned:
            result[s] = "NOT_IN_OUTPUT"
        else:
            try:
                df = df_out.xs(s, level=1, axis=1).dropna(how="all")
                result[s] = "ok" if not df.empty else "ALL_NaN"
            except KeyError:
                result[s] = "NOT_IN_OUTPUT"
    return result

# ── run ───────────────────────────────────────────────────────────────────────────
batch_stats = []
all_results: dict[str, str] = {}
t_start = time.time()

sym_pbar = tqdm(total=len(ALL_SYMBOLS), desc="symbols", unit="sym", position=0)
batch_pbar = tqdm(total=len(BATCHES),   desc="batches", unit="batch", position=1)

for bi, batch in enumerate(BATCHES):
    if bi > 0:
        time.sleep(BATCH_SLEEP)

    t0 = time.time()
    res = download_batch(batch)
    elapsed = time.time() - t0

    fail_breakdown: dict[str, int] = defaultdict(int)
    ok_count = 0
    for sym, status in res.items():
        all_results[sym] = status
        if status == "ok":
            ok_count += 1
        else:
            fail_breakdown[status] += 1

    sym_pbar.update(len(batch))
    sym_pbar.set_postfix(ok=sum(1 for v in all_results.values() if v == "ok"),
                         fail=sum(1 for v in all_results.values() if v != "ok"))
    batch_pbar.update(1)
    batch_pbar.set_postfix(elapsed=f"{elapsed:.1f}s", ok=f"{ok_count}/{len(batch)}")

    batch_stats.append({
        "batch":   bi + 1,
        "n":       len(batch),
        "elapsed": elapsed,
        "ok":      ok_count,
        "fails":   dict(fail_breakdown),
    })

sym_pbar.close()
batch_pbar.close()
total_elapsed = time.time() - t_start

# ── report ────────────────────────────────────────────────────────────────────────
total_symbols = len(ALL_SYMBOLS)
total_ok      = sum(1 for s in all_results.values() if s == "ok")
fail_summary: dict[str, int] = defaultdict(int)
for s in all_results.values():
    if s != "ok":
        fail_summary[s] += 1

lines: list[str] = []

def w(line: str = ""):
    lines.append(line)
    print(line)

w("=" * 65)
w("  threads=True  全量 SYMBOL 下载速度测试")
w(f"  date range : {START} → {END}   |   run : {date.today()}")
w("=" * 65)

w()
w("── TIMING ──────────────────────────────────────────────────────")
w(f"  total symbols   : {total_symbols}")
w(f"  total batches   : {len(BATCHES)}  (batch_size={BATCH_SIZE})")
w(f"  total elapsed   : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
w(f"  per symbol      : {total_elapsed/total_symbols*1000:.1f} ms")
w(f"  throughput      : {total_symbols/total_elapsed:.1f} symbols/s")

w()
w("── COVERAGE ────────────────────────────────────────────────────")
w(f"  success (ok)    : {total_ok} / {total_symbols}  ({total_ok/total_symbols*100:.1f}%)")
w(f"  failed          : {total_symbols - total_ok}")

if fail_summary:
    w()
    w("── FAILURE BREAKDOWN ───────────────────────────────────────────")
    for tag, cnt in sorted(fail_summary.items(), key=lambda x: -x[1]):
        w(f"  {tag:<22}  {cnt:>6}")

w()
w("── BATCH-LEVEL BREAKDOWN ───────────────────────────────────────")
w(f"  {'batch':>5}  {'n':>5}  {'ok':>6}  {'ok%':>5}  {'elapsed':>8}  fails")
for bs in batch_stats:
    pct = bs["ok"] / bs["n"] * 100
    fail_str = "  ".join(f"{k}:{v}" for k, v in bs["fails"].items()) if bs["fails"] else "-"
    w(f"  {bs['batch']:>5}  {bs['n']:>5}  {bs['ok']:>6}  {pct:>4.0f}%  {bs['elapsed']:>7.1f}s  {fail_str}")

w()
w("=" * 65)

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved → {REPORT_PATH.resolve()}")
