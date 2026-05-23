"""
Turtle-style intraday breakout backtest.

For each symbol with minute data on target_date:
  1. Load daily history → compute prior `lookback_days` high (excluding target_date)
  2. Run turtle policy on minute bars:
     - entry on close > daily_high
     - pyramid +step_pct, up to n_max tranches (each a fixed cash slice)
     - stop = prev tranche price (or init_stop_pct below entry for first tranche)
     - one entry/exit cycle per day
  3. Aggregate pnl across all symbols.

Usage:
    uv run research/turtle_intraday_bt.py
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings
import logging
import contextlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("gym").setLevel(logging.CRITICAL)


@contextlib.contextmanager
def _suppress():
    with open(os.devnull, "w") as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


with _suppress():
    import gym  # noqa: F401

from gym_trade.env.embodied import PaperTrade
import gym_trade.policy  # triggers auto-register
from gym_trade.policy.registry import POLICY_REGISTRY
from gym_trade.tool.get_data import load_data as load_data_func
from gym_trade.tool.screener import get_symbols_from_minute_dir, screen_universe


REPO_ROOT = Path(__file__).resolve().parent.parent


def _daily_cache_dir(repo_root: Path) -> Path:
    return repo_root / ".cache" / "us-stock_1d_yfinance"


def _load_daily_csv(cache_dir: Path, symbol: str) -> pd.DataFrame | None:
    hits = list(cache_dir.glob(f"{symbol}_????-??-??_????-??-??.csv"))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def _load_all_daily(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for s in tqdm(symbols, desc="loading daily cache"):
        df = _load_daily_csv(cache_dir, s)
        if df is not None and len(df) > 0:
            out[s] = df
    return out


def _daily_close_on(daily_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    idx = daily_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    rows = daily_df.loc[idx.normalize() == date.normalize()]
    if len(rows) == 0:
        return None
    return float(rows["close"].iloc[0])


def _minute_close_eod(minute_df: pd.DataFrame) -> float | None:
    """Last regular-session close from the minute df. Falls back to last row."""
    if len(minute_df) == 0:
        return None
    # last regular bar is the very last row (env's eod_close handles boundary already)
    return float(minute_df["close"].iloc[-1])


def _compute_prior_high(daily_df: pd.DataFrame, target_date: pd.Timestamp, lookback_days: int,
                         scale: float = 1.0) -> float | None:
    """Prior `lookback_days` daily high, excluding target_date. Multiplied by `scale`
    to undo yfinance auto-adjust (splits/dividends) so it lines up with raw minute prices."""
    idx = daily_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    mask = idx.normalize() < target_date.normalize()
    prior = daily_df.loc[mask]
    if len(prior) < lookback_days:
        return None
    window = prior.tail(lookback_days)
    return float(window["high"].max()) * scale


def _compute_prev_close(daily_df: pd.DataFrame, target_date: pd.Timestamp,
                         scale: float = 1.0) -> float | None:
    idx = daily_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    prior = daily_df.loc[idx.normalize() < target_date.normalize()]
    if len(prior) == 0:
        return None
    return float(prior["close"].iloc[-1]) * scale


def _adjust_scale(daily_df: pd.DataFrame, minute_df: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    """Split/dividend correction factor: raw_minute_close_today / adjusted_daily_close_today.
    All daily prices on/before target_date carry the same compounded post-target adjustments,
    so this single ratio rescales the whole prior daily window to the minute-data price scale."""
    d_close = _daily_close_on(daily_df, target_date)
    m_close = _minute_close_eod(minute_df)
    if d_close is None or m_close is None or d_close <= 0:
        return None
    return m_close / d_close


@dataclass
class SymResult:
    symbol: str
    pnl: float                 # pnl pct relative to init_balance
    n_tranches: int            # peak number of tranches held
    entry_count: int           # number of buy events
    exit_count: int            # number of exits (0 or 1)
    breakout_hit: bool         # whether close ever crossed the daily high
    gain_hit: bool             # whether close ever cleared prev_close * (1+gain)
    daily_high_prior: float
    prev_close: float
    bars: int


def _run_one(
    symbol: str,
    minute_df: pd.DataFrame,
    daily_high_prior: float,
    prev_close: float,
    init_balance: float,
    n_max: int,
    step_pct: float,
    initial_stop_pct: float,
    entry_gain_pct: float,
    commission_type: str,
) -> SymResult:
    df = minute_df.dropna(subset=["open", "close", "high", "low"]).copy()
    if len(df) < 2:
        return SymResult(symbol, 0.0, 0, 0, 0, False, False, daily_high_prior, prev_close, len(df))

    df["ta@d52h"] = daily_high_prior
    df["ta@prev_close"] = prev_close

    policy_cls = POLICY_REGISTRY["turtle_intraday"]
    policy = policy_cls()
    hp = {
        "step_pct": step_pct,
        "n_max": n_max,
        "initial_stop_pct": initial_stop_pct,
        "entry_gain_pct": entry_gain_pct,
        "init_balance": init_balance,
    }
    policy.set_hyper_param(hp)
    policy.init_policy()

    col_range_dict = {"ta@d52h": [0.0, np.inf], "ta@prev_close": [0.0, np.inf]}
    env = PaperTrade(
        df=df,
        interval="1m",
        obs_keys=policy.obs_keys,
        init_balance=init_balance,
        commission_type=commission_type,
        reward_type="sparse",
        dash_keys=["pos", "cash", "balance", "pnl"],
        action_on="open_t",
        eod_close=True,
        col_range_dict=col_range_dict,
    )
    obs = env.reset()
    done = False
    entry_count = 0
    exit_count = 0
    peak_tranches = 0
    gain_thresh_price = prev_close * (1.0 + entry_gain_pct) if prev_close > 0 else np.inf
    first_close = float(df["close"].iloc[0])
    breakout_hit = first_close > daily_high_prior
    gain_hit = first_close >= gain_thresh_price
    while not done:
        action, info = policy(obs)
        if info["entry_point"]:
            entry_count += 1
        if info["exit_point"]:
            exit_count += 1
        peak_tranches = max(peak_tranches, len(policy._tranche_prices))
        obs, _, done, _ = env.step(action)
        c_now = float(obs.get("close", 0.0))
        if not breakout_hit and c_now > daily_high_prior:
            breakout_hit = True
        if not gain_hit and c_now >= gain_thresh_price:
            gain_hit = True

    return SymResult(
        symbol=symbol,
        pnl=float(env.pnl),
        n_tranches=peak_tranches,
        entry_count=entry_count,
        exit_count=exit_count,
        breakout_hit=breakout_hit,
        gain_hit=gain_hit,
        daily_high_prior=daily_high_prior,
        prev_close=prev_close,
        bars=len(df),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_date", default="2022-02-18")
    ap.add_argument("--minute_dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--lookback_days", type=int, default=52)
    ap.add_argument("--n_max", type=int, default=5)
    ap.add_argument("--step_pct", type=float, default=0.05)
    ap.add_argument("--initial_stop_pct", type=float, default=0.03)
    ap.add_argument("--entry_gain_pct", type=float, default=0.10,
                    help="entry requires close >= prev_close * (1 + this)")
    ap.add_argument("--init_balance", type=float, default=1e6)
    ap.add_argument("--commission_type", default="free", choices=["free", "futu"])
    ap.add_argument("--min_price", type=float, default=5.0)
    ap.add_argument("--min_adv", type=float, default=10_000_000)
    ap.add_argument("--max_symbols", type=int, default=0, help="0 = no limit")
    ap.add_argument("--save_csv", default=str(REPO_ROOT / "outputs" / "turtle_intraday_bt.csv"))
    args = ap.parse_args()

    target_date = pd.Timestamp(args.target_date)
    print(f"[turtle_bt] target_date={args.target_date} lookback={args.lookback_days}d "
          f"n_max={args.n_max} step={args.step_pct:.1%} stop={args.initial_stop_pct:.1%} "
          f"entry_gain={args.entry_gain_pct:.1%}")

    # 1. enumerate symbols with minute data
    all_syms = get_symbols_from_minute_dir(args.minute_dir, args.target_date)
    print(f"[turtle_bt] {len(all_syms)} symbols have minute data on {args.target_date}")

    # 2. load daily cache (no network)
    cache_dir = _daily_cache_dir(REPO_ROOT)
    daily_dfs = _load_all_daily(all_syms, cache_dir)
    print(f"[turtle_bt] daily history loaded for {len(daily_dfs)} symbols")

    # 3. universe filter (price + ADV) using daily history through target_date
    universe = screen_universe(daily_dfs, args.target_date, min_price=args.min_price,
                               min_avg_dollar_volume=args.min_adv)
    print(f"[turtle_bt] {len(universe)} symbols pass universe filter "
          f"(close>${args.min_price}, ADV>${args.min_adv/1e6:.0f}M)")

    if args.max_symbols > 0:
        universe = universe[: args.max_symbols]
        print(f"[turtle_bt] truncated to {len(universe)} symbols (max_symbols)")

    # 4. load minute data first (needed to derive split-adjust scale per symbol)
    next_day = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[turtle_bt] loading minute data for {len(universe)} symbols ...")
    minute_dfs = load_data_func(
        data_api="local",
        interval="1m",
        symbols=universe,
        local_data_dir=args.minute_dir,
        start=args.target_date,
        end=next_day,
        proxy=None,
    )
    print(f"[turtle_bt] minute data ready for {len(minute_dfs)} symbols")

    # 5. compute scale, prior 52-day high & prev-day close per symbol
    sym_highs: dict[str, float] = {}
    sym_prev_close: dict[str, float] = {}
    sym_scale: dict[str, float] = {}
    n_skip_scale = 0
    for s, mdf in minute_dfs.items():
        if s not in daily_dfs:
            continue
        scale = _adjust_scale(daily_dfs[s], mdf, target_date)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            n_skip_scale += 1
            continue
        h = _compute_prior_high(daily_dfs[s], target_date, args.lookback_days, scale=scale)
        pc = _compute_prev_close(daily_dfs[s], target_date, scale=scale)
        if h is not None and pc is not None and np.isfinite(h) and np.isfinite(pc) and pc > 0:
            sym_highs[s] = h
            sym_prev_close[s] = pc
            sym_scale[s] = scale
    print(f"[turtle_bt] {len(sym_highs)} symbols have ≥{args.lookback_days} prior daily bars + prev close "
          f"(skipped {n_skip_scale} with no scale)")
    if sym_scale:
        scales = np.array(list(sym_scale.values()))
        print(f"[turtle_bt] split-adjust scale: min={scales.min():.3f} med={np.median(scales):.3f} "
              f"max={scales.max():.3f}  (>1 means symbol had splits since target_date)")

    if not sym_highs:
        print("[turtle_bt] nothing to backtest.")
        return

    # 6. backtest each symbol (only those with valid scale + history)
    results: list[SymResult] = []
    for sym in tqdm(sorted(sym_highs.keys()), desc="backtest"):
        try:
            r = _run_one(
                symbol=sym,
                minute_df=minute_dfs[sym],
                daily_high_prior=sym_highs[sym],
                prev_close=sym_prev_close[sym],
                init_balance=args.init_balance,
                n_max=args.n_max,
                step_pct=args.step_pct,
                initial_stop_pct=args.initial_stop_pct,
                entry_gain_pct=args.entry_gain_pct,
                commission_type=args.commission_type,
            )
            results.append(r)
        except Exception as e:
            print(f"  [skip] {sym}: {type(e).__name__}: {e}")

    if not results:
        print("[turtle_bt] no results.")
        return

    # 7. aggregate
    res_df = pd.DataFrame([asdict(r) for r in results])
    traded = res_df[res_df["entry_count"] > 0].copy()

    print("\n=== Aggregate (all backtested symbols) ===")
    print(f"  symbols backtested : {len(res_df)}")
    print(f"  breakout occurred  : {int(res_df['breakout_hit'].sum())}")
    print(f"  gain≥{args.entry_gain_pct:.1%} hit    : {int(res_df['gain_hit'].sum())}")
    print(f"  both conditions     : {int((res_df['breakout_hit'] & res_df['gain_hit']).sum())}")
    print(f"  entered a trade    : {len(traded)}")
    print(f"  avg pnl (all)      : {res_df['pnl'].mean()*100:+.3f} %")
    print(f"  sum  pnl (all)     : {res_df['pnl'].sum()*100:+.3f} %  (sum of per-symbol % returns)")

    if len(traded) > 0:
        wins = traded[traded["pnl"] > 0]
        losses = traded[traded["pnl"] < 0]
        print("\n=== Among symbols that entered ===")
        print(f"  trades         : {len(traded)}")
        print(f"  win  : {len(wins)} ({len(wins)/len(traded)*100:.1f}%)")
        print(f"  loss : {len(losses)} ({len(losses)/len(traded)*100:.1f}%)")
        print(f"  avg pnl     : {traded['pnl'].mean()*100:+.3f} %")
        print(f"  med pnl     : {traded['pnl'].median()*100:+.3f} %")
        print(f"  best        : {traded['pnl'].max()*100:+.3f} %  ({traded.loc[traded['pnl'].idxmax(),'symbol']})")
        print(f"  worst       : {traded['pnl'].min()*100:+.3f} %  ({traded.loc[traded['pnl'].idxmin(),'symbol']})")
        if len(wins) > 0:
            print(f"  avg win     : {wins['pnl'].mean()*100:+.3f} %")
        if len(losses) > 0:
            print(f"  avg loss    : {losses['pnl'].mean()*100:+.3f} %")
        avg_tranches = traded["n_tranches"].mean()
        print(f"  avg peak tranches : {avg_tranches:.2f} / {args.n_max}")

        # equal-cash portfolio: each symbol holds init_balance, return = mean(pnl)
        port_ret = traded["pnl"].mean()
        print(f"\n  if you spread the same $ across all {len(traded)} traded symbols,")
        print(f"  portfolio return on {args.target_date} ≈ {port_ret*100:+.3f} %")

    # 8. save
    out_path = Path(args.save_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.sort_values("pnl", ascending=False).to_csv(out_path, index=False)
    print(f"\n[turtle_bt] per-symbol results saved → {out_path}")

    # top winners / losers
    top = res_df[res_df["entry_count"] > 0].sort_values("pnl", ascending=False)
    if len(top) > 0:
        print("\n  top winners:")
        for _, r in top.head(10).iterrows():
            print(f"    {r['symbol']:8s}  pnl={r['pnl']*100:+7.3f}%  tranches={int(r['n_tranches'])}/{args.n_max}  exit={int(r['exit_count'])}")
        print("\n  top losers:")
        for _, r in top.tail(10).iterrows():
            print(f"    {r['symbol']:8s}  pnl={r['pnl']*100:+7.3f}%  tranches={int(r['n_tranches'])}/{args.n_max}  exit={int(r['exit_count'])}")


if __name__ == "__main__":
    main()
