"""
Screener: filters symbols from a daily OHLCV dict based on various criteria.
Each screener function takes {symbol: daily_df} and a target_date, returns
a sorted list of (symbol, metric_value) tuples.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def screen_universe(
    daily_dfs: dict[str, pd.DataFrame],
    target_date: str,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
    adv_window: int = 20,
) -> list[str]:
    """
    Step 1 basic universe filter — run daily before any strategy screen.

    Filters:
        - close price on target_date  > min_price          (default $5)
        - avg daily dollar volume over adv_window days
          ending on target_date       > min_avg_dollar_volume  (default $10 M)

    Dollar volume = close * volume (each day).

    Returns a sorted list of symbols that pass all filters.
    """
    target_ts = pd.Timestamp(target_date).normalize()
    passed: list[str] = []

    for symbol, df in daily_dfs.items():
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        idx_norm = idx.normalize()

        # rows up to and including target_date
        mask_le = idx_norm <= target_ts
        if not mask_le.any():
            continue

        sub = df.loc[mask_le]

        # ── price filter ────────────────────────────────────────────────────
        close_today = float(sub["close"].iloc[-1])
        if np.isnan(close_today) or close_today <= min_price:
            continue

        # ── avg daily dollar volume filter ──────────────────────────────────
        window = sub.tail(adv_window)
        dollar_vol = window["close"] * window["volume"]
        adv = float(dollar_vol.mean())
        if np.isnan(adv) or adv <= min_avg_dollar_volume:
            continue

        passed.append(symbol)

    return sorted(passed)


def screen_universe_from_minute(
    minute_dir: str,
    target_date: str,
    min_price: float = 5.0,
    min_dollar_volume: float = 10_000_000,
) -> list[str]:
    """
    Fast universe filter using only local minute data — no network required.

    Reads each YYYY-MM-DD-SYMBOL.csv for target_date, computes:
      - price        : close price of the last bar of the session
      - dollar_volume: sum(close * volume) over all bars that day

    Returns sorted list of symbols that pass both filters.
    """
    date_str = pd.Timestamp(target_date).strftime('%Y-%m-%d')
    passed: list[str] = []

    for f in sorted(Path(minute_dir).rglob("*.csv")):
        parts = f.stem.split('-', 3)
        if len(parts) != 4 or f"{parts[0]}-{parts[1]}-{parts[2]}" != date_str:
            continue
        symbol = parts[3]
        try:
            df = pd.read_csv(str(f), usecols=['Datetime', 'Close', 'Volume'])
            df = df.dropna(subset=['Close', 'Volume'])
            if len(df) == 0:
                continue
            price = float(df['Close'].iloc[-1])
            if np.isnan(price) or price <= min_price:
                continue
            dollar_vol = float((df['Close'] * df['Volume']).sum())
            if np.isnan(dollar_vol) or dollar_vol <= min_dollar_volume:
                continue
            passed.append(symbol)
        except Exception:
            continue

    return sorted(passed)


def get_symbols_from_minute_dir(minute_dir: str, target_date: str) -> list[str]:
    """
    Recursively scan a minute data directory and return all symbols whose files
    match target_date.  File naming: YYYY-MM-DD-SYMBOL.csv (inside any subfolder).
    """
    date_str = pd.Timestamp(target_date).strftime('%Y-%m-%d')
    symbols = []
    for f in sorted(Path(minute_dir).rglob("*.csv")):
        parts = f.stem.split('-', 3)
        if len(parts) == 4 and f"{parts[0]}-{parts[1]}-{parts[2]}" == date_str:
            symbols.append(parts[3])
    return symbols


def screen_gap(
    daily_dfs: dict[str, pd.DataFrame],
    target_date: str,
    top_n: int = 10,
    direction: str = "up",       # "up" or "down"
    min_gap_pct: float = 0.0,    # minimum absolute gap to qualify
) -> list[dict]:
    """
    Screen symbols by overnight gap on target_date.
    gap = open[target_date] / close[prev_trading_day] - 1

    Returns a list of dicts sorted by gap descending (for "up") or ascending (for "down"),
    limited to top_n results.
    """
    assert direction in ("up", "down")
    target_ts = pd.Timestamp(target_date)
    results = []

    for symbol, df in daily_dfs.items():
        # Normalise index to timezone-naive date for comparison
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_localize(None)
        idx_dates = idx.normalize()

        # Find target date row
        mask = idx_dates == target_ts.normalize()
        if not mask.any():
            continue
        target_pos = int(np.where(mask)[0][0])
        if target_pos == 0:
            continue  # no previous day

        open_today  = float(df['open'].iloc[target_pos])
        close_prev  = float(df['close'].iloc[target_pos - 1])
        if close_prev == 0 or np.isnan(close_prev) or np.isnan(open_today):
            continue

        gap = open_today / close_prev - 1.0

        if direction == "up"   and gap < min_gap_pct:
            continue
        if direction == "down" and gap > -min_gap_pct:
            continue

        results.append({
            "symbol":     symbol,
            "gap":        gap,
            "open_today": open_today,
            "close_prev": close_prev,
        })

    reverse = (direction == "up")
    results.sort(key=lambda x: x["gap"], reverse=reverse)
    return results[:top_n]
