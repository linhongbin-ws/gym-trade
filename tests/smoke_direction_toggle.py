"""Smoke test for the direction_toggle policy. Run from repo root:
   .venv/bin/python tests/smoke_direction_toggle.py
"""
import contextlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
import gym_trade.policy  # auto-discovery
from gym_trade.policy.registry import POLICY_REGISTRY, FUNCTION_REGISTRY


def main():
    # 1. registry presence
    assert "direction_toggle" in POLICY_REGISTRY, f"policy not registered. keys: {list(POLICY_REGISTRY)}"
    assert "direction_toggle@features" in FUNCTION_REGISTRY, f"features not registered. keys: {list(FUNCTION_REGISTRY)}"
    print(f"[ok] registry: policy={list(POLICY_REGISTRY)}  funcs={list(FUNCTION_REGISTRY)}")

    # 2. load a daily df
    csv = Path(__file__).resolve().parent.parent / "gym_trade" / "asset" / "mini_daily" / "AAPL.csv"
    df = pd.read_csv(csv, parse_dates=["Date"]).rename(columns=str.lower).set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.iloc[:200]  # keep small — direction_toggle is O(N²)
    print(f"[ok] loaded {csv.name}  bars={len(df)}  range={df.index[0].date()}→{df.index[-1].date()}")

    # 3. features
    features_fn = FUNCTION_REGISTRY["direction_toggle@features"]
    feat_df, col_range = features_fn(df)
    for col in feat_df.columns:
        df[col] = feat_df[col]
    print(f"[ok] features: cols={list(feat_df.columns)}  "
          f"acc[min,max]=[{feat_df['ta@dt_strongup_acc'].min()},{feat_df['ta@dt_strongup_acc'].max()}]")

    # 4. construct policy and env
    policy_cls = POLICY_REGISTRY["direction_toggle"]
    policy = policy_cls()
    policy.set_hyper_param(policy.randomize_hyper_param(random_type=None))  # use defaults
    policy.init_policy()

    env = PaperTrade(
        df=df,
        interval="1d",
        obs_keys=policy.obs_keys,
        init_balance=1e6,
        commission_type="free",
        reward_type="sparse",
        dash_keys=["pos", "cash", "balance", "pnl"],
        action_on="open_t",
        eod_close=False,
        col_range_dict=col_range.copy(),
    )

    obs = env.reset()
    entries, exits = 0, 0
    done = False
    while not done:
        action, info = policy(obs)
        if info["entry_point"]:
            entries += 1
        if info["exit_point"]:
            exits += 1
        obs, _, done, _ = env.step(action)

    print(f"[ok] rollout: entries={entries}  exits={exits}  pnl={env.pnl:+.4f}  bars={env._t + 1}")

    # 5. quick hyperparam search check
    samples = [policy.randomize_hyper_param(random_type="uniform") for _ in range(3)]
    assert all("sig_cnt_thres" in s and "entry_pos_thres" in s and "exit_pos_thres" in s for s in samples)
    print(f"[ok] hyper samples: {samples}")

    print("\nALL OK")


if __name__ == "__main__":
    main()
