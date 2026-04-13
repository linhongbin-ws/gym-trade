
import os
import sys
import warnings
import logging
import contextlib

warnings.filterwarnings("ignore")
logging.getLogger("gym").setLevel(logging.CRITICAL)


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# disable gym warning
with suppress_output():
    import gym


from dataclasses import dataclass
from gym_trade.tool.get_data import load_data as load_data_func
from gym_trade.tool.lw_chart import ChartMod as Chart
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from gym_trade.env.embodied import PaperTrade
from datetime import datetime, timedelta
from pathlib import Path
import gym_trade.policy  # triggers auto-discovery via __init__.py
from gym_trade.policy.registry import POLICY_REGISTRY, FUNCTION_REGISTRY
from gym_trade.tool.screener import get_symbols_from_minute_dir, screen_gap, screen_universe, screen_universe_from_minute
from tqdm import tqdm
import yaml
from queue import Empty as QEmpty
import multiprocessing as mp
from typing import Any


def _ensure_daily_cache(
    symbols: list[str],
    target_date: str,
    proxy: str | None,
    cache_dir: str,
    market: str = "us-stock",
    interval: str = "1d",
    data_api: str = "yfinance",
    cache_only: bool = False,
    lookback_years: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Ensure daily data cache covers [target_date - lookback_years, target_date] for every symbol.

    - Cache exists AND covers the needed window → load as-is, no download.
    - Cache missing or doesn't cover → delete stale file and re-download.
    Download range: [need_start, target_date].
    """
    cache_csv_dir = Path(cache_dir) / f"{market}_{interval}_{data_api}"
    cache_csv_dir.mkdir(parents=True, exist_ok=True)

    need_start = pd.Timestamp(target_date) - pd.DateOffset(years=lookback_years)
    need_end   = pd.Timestamp(target_date)
    dl_start   = need_start.strftime('%Y-%m-%d')
    dl_end     = (need_end + pd.DateOffset(days=1)).strftime('%Y-%m-%d')  # yfinance end is exclusive

    def _find_cache(s: str) -> Path | None:
        hits = list(cache_csv_dir.glob(f"{s}_????-??-??_????-??-??.csv"))
        return hits[0] if hits else None

    def _cache_covers(path: Path) -> bool:
        """Parse dates from filename SYMBOL_start_end.csv and check coverage.
        Allow 7-day tolerance on start (markets may not trade on exact need_start).
        """
        try:
            parts = path.stem.rsplit('_', 2)   # ['SYMBOL', 'YYYY-MM-DD', 'YYYY-MM-DD']
            first = pd.Timestamp(parts[1])
            last  = pd.Timestamp(parts[2])
            return first <= need_start + pd.DateOffset(days=7) and last >= need_end - pd.DateOffset(days=7)
        except Exception:
            return False

    stale: list[str] = []
    for s in symbols:
        cf = _find_cache(s)
        if cf is None or not _cache_covers(cf):
            stale.append(s)

    cached = len(symbols) - len(stale)
    print(f"[screen_bt] cache hit: {cached} / {len(symbols)}  |  need download: {len(stale)} "
          f"(range: {dl_start} ~ {dl_end})")

    if stale and not cache_only:
        for s in stale:
            cf = _find_cache(s)
            if cf is not None:
                cf.unlink(missing_ok=True)
    elif stale and cache_only:
        print(f"[screen_bt] cache_only=true, skipping {len(stale)} missing/stale symbols")

    return load_data_func(
        data_api=data_api,
        proxy=proxy,
        interval=interval,
        start=dl_start,
        end=dl_end,
        symbols=symbols,
        cache_dir=cache_dir,
        market=market,
        cache_only=cache_only,
    )


def gen_stat(key,values):
    stat = {}
    stat[key + "_mean"] = np.mean(values)
    stat[key + "_std"] = np.std(values)
    stat[key + "_min"] = np.min(values)
    stat[key + "_max"] = np.max(values)
    stat[key + "_median"] = np.median(values)

    return stat


def df_generator(dfs: dict[str, pd.DataFrame]):
    loop_id = 0
    while True:
        for k, df in dfs.items():
            yield k,df, loop_id
        loop_id +=1



def deal_with_best_pnl(best_pnl_stat: dict, result_list: list,  pbar_search: tqdm, save_result_dir: Path, file_name: str):
    pnl_stat = {}
    pnl_stat.update(gen_stat("pnl", np.array([result.pnl for result in result_list])))
    pnl_stat.update(gen_stat("pos_chg", np.array([result.pos_chg for result in result_list])))
    pnl_stat.update(gen_stat("hold_t", np.array([result.hold_t for result in result_list])))
    pnl_stat.update(gen_stat("total_t", np.array([result.total_t for result in result_list])))
    pnl_stat['policy_hyper_param'] = result_list[0].policy_hyper_param

    if best_pnl_stat is None:
        best_pnl_stat = pnl_stat
    elif pnl_stat["pnl_mean"] > best_pnl_stat["pnl_mean"]:
        best_pnl_stat = pnl_stat
    else:
        return best_pnl_stat

    pbar_search.write(
        f"best pnl mean: {best_pnl_stat['pnl_mean']:.3f}, pos chg mean: {best_pnl_stat['pos_chg_mean']} ,"
        f" hold t mean: {best_pnl_stat['hold_t_mean']} / total t mean {best_pnl_stat['total_t_mean']} "
    )
    save_result_dir = Path(save_result_dir)
    save_result_dir.mkdir(parents=True, exist_ok=True)
    file = save_result_dir / file_name
    best_pnl_stat_save = {"best_pnl": best_pnl_stat}
    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(
            to_python(best_pnl_stat_save),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=4,
        )
    return best_pnl_stat

def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    else:
        return obj


def _resolve_cache_dir(cfg: DictConfig) -> str:
    orig_cwd = hydra.utils.get_original_cwd()
    return str(Path(orig_cwd) / cfg.data.get('cache_dir', '.cache'))


def load_data(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    dfs = load_data_func(
        data_api=cfg.data.name,
        proxy=cfg.general.proxy,
        interval=cfg.data.interval,
        start=cfg.data.start,
        end=cfg.data.end,
        symbols=cfg.data.symbol,
        cache_dir=_resolve_cache_dir(cfg),
        market=cfg.data.get('market', 'us-stock'),
        force_download=cfg.data.get('force_download', False),
        cache_only=cfg.data.get('cache_only', False),
        local_data_dir=cfg.data.get('local_dir', None),
    )
    return dfs


def make_ta_features(cfg: DictConfig, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    print("making features...")
    ft_name = cfg.policy.name + "@features"
    assert ft_name in FUNCTION_REGISTRY, f"Feature {ft_name} not found, select from {FUNCTION_REGISTRY.keys()}"
    func_call = FUNCTION_REGISTRY[ft_name]
    feature_params = OmegaConf.to_container(cfg.get("features", {}), resolve=True)
    col_range_dict = {}
    pbar = tqdm(total=len(dfs), desc="making features")
    for sym, df in dfs.items():
        pbar.update(1)
        feat_df, col_range_dict = func_call(df, **feature_params)
        for col in feat_df.columns:
            df[col] = feat_df[col]
    return dfs, col_range_dict


@dataclass
class BTRequest:
    policy_hyper_param: dict
    df: pd.DataFrame
    param_id: int
    df_name: str


@dataclass
class BTResult:
    pnl: float
    policy_hyper_param: dict
    pos_chg: int
    hold_t: int
    total_t: int
    param_id: int
    entry_dates: list[pd.DatetimeIndex]
    exit_dates: list[pd.DatetimeIndex]
    df_name: str


def bt_rollout(request: BTRequest, policy, env: PaperTrade, stop_event=None):
    policy.set_hyper_param(request.policy_hyper_param)
    policy.init_policy()
    obs = env.reset()
    done = False
    pos_prv = obs["dash@pos"]
    pos_chg = 0
    hold_cnt = 0
    total_t = 0
    entry_dates = []
    exit_dates = []
    while not done:
        if stop_event is not None:
            if stop_event.is_set():
                break
        total_t += 1
        action, action_info = policy(obs)
        if action_info["entry_point"]:
            entry_dates.append(env._t)
        if action_info["exit_point"]:
            exit_dates.append(env._t)
        obs, reward, done, info = env.step(action)
        if obs["dash@pos"] != pos_prv:
            pos_chg += 1
        if obs["dash@pos"] > 0:
            hold_cnt += 1
        # if env._t % 500 == 0:
        # print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
        pos_prv = obs["dash@pos"]

    result = BTResult(
        pnl=env.pnl,
        policy_hyper_param=policy.hyper_param,
        pos_chg=pos_chg,
        param_id=request.param_id,
        hold_t=hold_cnt,
        total_t=total_t,
        entry_dates=entry_dates,
        exit_dates=exit_dates,
        df_name=request.df_name,
    )
    return result


def bt_server_loop(
    policy_name, policy_args, env_args, stop_event, request_queue, result_queue
):
    policy_cls = POLICY_REGISTRY[policy_name]
    policy = policy_cls(**policy_args)

    while not stop_event.is_set():
        request = None
        try:
            request = request_queue.get(timeout=0.5)  # 定期醒来检查 stop_event
        except QEmpty:
            continue
        if request is not None:
            env = PaperTrade(df = request.df, **env_args)
            result = bt_rollout(request, policy, env, stop_event)
            result_queue.put(result)


class BTServer:
    """recieve a bt request, and return a result"""

    def __init__(
        self,
        cfg: DictConfig,
        policy_name: str,
        policy_args: dict[str, Any],
        env_args: dict[str, Any],
        n_workers: int = 2,
    ):
        self._n_workers = cfg.mode.workers

        if self._n_workers > 1:
            ctx = mp.get_context("spawn")  # 跨平台更稳（Windows/macOS 必须 spawn）
            self._request_queue = ctx.Queue(maxsize=self.n_workers * 4)
            self._result_queue = ctx.Queue()
            self._stop_event = ctx.Event()

            self.procs = [
                ctx.Process(
                    target=bt_server_loop,
                    args=(
                        policy_name,
                        policy_args,
                        env_args,
                        self._stop_event,
                        self._request_queue,
                        self._result_queue,
                    ),
                )
                for _ in range(self.n_workers)
            ]
            for p in self.procs:
                # p.daemon = True
                p.start()

        self._closed = False
        self._cfg = cfg
        self._policy_args = policy_args
        self._env_args = env_args
        self._policy_name = policy_name

    @property
    def n_workers(self):
        return self._n_workers

    def shutdown(self, join_timeout: float = 5.0):

        if self._n_workers > 1:
            self._stop_event.set()

            # wait for graceful exit
            for p in self.procs:
                p.join(join_timeout)

            # force kill remaining
            for p in self.procs:
                if p.is_alive():
                    p.terminate()
                    p.join()

            # NOW it's safe to close queues
            self._request_queue.close()
            self._result_queue.close()
            self._request_queue.join_thread()
            self._result_queue.join_thread()

    # 让 with BTServer(...) 自动清理
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    def backtest(self, dfs: dict[str, pd.DataFrame]):

        policy_cls = POLICY_REGISTRY[self._cfg.policy.name]
        policy = policy_cls(**self._policy_args)
        best_pnl_stat = None
        file_name = "bt_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".yaml"

 
        
        
        pbar_search = tqdm(
            total=self._cfg.mode.search_num,
            desc="Search",
            position=0
        )

       
        # for df_idx in range(len(df_list)):
        pbar_df = tqdm(
            total=len(dfs),
            desc="DF",
            position=1,
            leave=False   # 内层结束自动消失（更干净）
        )
        df_gen = df_generator(dfs)
        result_dict = {i: [] for i in range(self._cfg.mode.search_num)}
        policy_params = [policy.randomize_hyper_param(
                        random_type=self._cfg.mode.hyper_search) for i in range(self._cfg.mode.search_num)]
        stop_gen =False
        while(
                not self._stop_event.is_set() 
                and pbar_search.n < self._cfg.mode.search_num
            ):


            if not self._request_queue.full() and not stop_gen:
                
                df_name,df, param_id = next(df_gen)
                if param_id < self._cfg.mode.search_num:
                    request = BTRequest(
                        policy_hyper_param=policy_params[param_id], param_id=param_id, df=df, df_name=df_name
                    )
                    self._request_queue.put(request)
                else:
                    stop_gen = True

            if not self._result_queue.empty():
                try:
                    result = self._result_queue.get(
                        timeout=0.01
                    )  # 定期醒来检查 stop_event
          
                    result_dict[result.param_id].append(result)
                except QEmpty:
                    pass
            

            now_search_idx = pbar_search.n
            current = len(result_dict[now_search_idx])

            if current == len(dfs):
                best_pnl_stat = deal_with_best_pnl(
                    best_pnl_stat,
                    result_dict[now_search_idx],
                    pbar_search,
                    self._cfg.mode.save_result_dir,
                    file_name
                )

                result_dict.pop(now_search_idx)

                pbar_search.update(1)
                pbar_df.reset()

            elif current < len(dfs):
                delta = current - pbar_df.n
                if delta > 0:
                    pbar_df.update(delta)

            else:
                raise ValueError(...)




def vis_lightweight_chart_df(
    chart: Chart,
    df: pd.DataFrame,
    df_name: str,
    labels: list[tuple[str, pd.DatetimeIndex]],
    entry_dates: list[pd.DatetimeIndex],
    exit_dates: list[pd.DatetimeIndex],
    mainchart_keys: list[str] = [],
    subchart_keys: list[str] = [],
    mainchart_height: float = 0.6,
):

    _df = df[["close", "open", "high", "low", "volume"]]
    _df["time"] = df.index
    chart.set(_df)
    def random_color():
            return f"rgba({random.randint(100, 255)}, {random.randint(100, 255)}, {random.randint(100, 255)}, 0.9)"
    


    lines = {}

    # assert False, mainchart_keys
    for k in mainchart_keys:
        line_df = pd.DataFrame({"time": df.index, k: df[k]})
        # line_df = line_df.dropna()
        lines[k] = chart.create_line(
            k,
            color=random_color(),
        )
        chart.legend(True)
        lines[k].set(line_df)

    for k in subchart_keys:
        subchart = chart.create_subchart(
            position="left",
            width=1,
            height=(1 - mainchart_height) / len(subchart_keys),
            sync=True,
        )
        subchart.legend(True)
        lines[k] = subchart.create_line(k)
        line_df = pd.DataFrame({"time": df.index, k: df[k]})
        # line_df = line_df.dropna()
        lines[k].set(line_df)

    # assert False, entry_dates
    for label, date in labels:
        chart.marker(text=label, time=date.to_pydatetime())
    # for entry_date in entry_dates[:3]:
    #     chart.marker(text="B", time=entry_date.to_pydatetime()) 
    # for exit_date in exit_dates[:3]:
    #     chart.marker(text="S", time=exit_date.to_pydatetime())
    

    chart.show(block=False)
    chart.press_n = False
    while True:
        if chart.press_n:
            break


def bt_mode(cfg: DictConfig, dfs: dict[str, pd.DataFrame], col_range_dict: dict) -> None:

    # create policy
    print(f"avaliable poliy {POLICY_REGISTRY.keys()}")
    policy_cls = POLICY_REGISTRY[cfg.policy.name]
    policy_args = {k: v for k, v in cfg.policy.items() if k not in ["name"]}
    policy = policy_cls(**policy_args)

    # create env
    env_args = OmegaConf.to_container(cfg.env, resolve=True)  # to dict
    env_args = {k: v for k, v in env_args.items() if k not in ["name", "start", "end"]}
    env_args["obs_keys"] = policy.obs_keys
    env_args["interval"] = cfg.data.interval
    env_args["col_range_dict"] = col_range_dict

    server = BTServer(
        cfg=cfg,
        policy_name=cfg.policy.name,
        policy_args=policy_args,
        env_args=env_args,
    )
    server.backtest(dfs)
    server.shutdown()


def screen_bt_mode(cfg: DictConfig) -> None:
    target_date = cfg.mode.target_date
    top_n = cfg.mode.get("top_n", 10)
    orig_cwd = hydra.utils.get_original_cwd()
    local_dir = str(Path(orig_cwd) / cfg.data.local_dir)

    min_price = cfg.mode.get("min_price", 5.0)
    min_adv   = cfg.mode.get("min_adv", 10_000_000)

    # 1. Get all symbols from minute data folder
    all_symbols = get_symbols_from_minute_dir(local_dir, target_date)
    print(f"[screen_bt] {len(all_symbols)} symbols found in minute data folder")

    # 2. Download daily data for all symbols (use cache if available)
    cache_dir = _resolve_cache_dir(cfg)
    cache_only = cfg.data.get("cache_only", False)
    daily_dfs = _ensure_daily_cache(
        symbols=all_symbols,
        target_date=target_date,
        proxy=cfg.general.proxy,
        cache_dir=cache_dir,
        market=cfg.data.get('market', 'us-stock'),
        interval="1d",
        data_api="yfinance",
        cache_only=cache_only,
    )
    print(f"[screen_bt] daily data loaded for {len(daily_dfs)} / {len(all_symbols)} symbols")

    # 3. Universe filter (20-day ADV window) using daily history
    universe  = screen_universe(daily_dfs, target_date, min_price=min_price, min_avg_dollar_volume=min_adv)
    daily_dfs = {s: daily_dfs[s] for s in universe if s in daily_dfs}
    print(f"[screen_bt] {len(daily_dfs)} symbols pass universe filter (price>${min_price}, adv>${min_adv/1e6:.0f}M)")

    # 4. Screen top-N gap-up symbols
    gap_results = screen_gap(daily_dfs, target_date, top_n=top_n, direction="up")
    print(f"\n[screen_bt] Top-{top_n} gap-up on {target_date}:")
    for r in gap_results:
        print(f"  {r['symbol']:10s}  gap={r['gap']*100:+.2f}%  open={r['open_today']:.2f}  prev_close={r['close_prev']:.2f}")

    if not gap_results:
        print("[screen_bt] No qualifying symbols found.")
        return

    top_symbols = [r["symbol"] for r in gap_results]

    # 5. Load minute data for top symbols on target_date only
    next_day = (pd.Timestamp(target_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    minute_dfs = load_data_func(
        data_api="local",
        interval="1m",
        symbols=top_symbols,
        local_data_dir=local_dir,
        start=target_date,
        end=next_day,
    )
    print(f"\n[screen_bt] Minute data loaded for {len(minute_dfs)} symbols")

    # 6. Run BNH on each symbol
    policy_cls = POLICY_REGISTRY["bnh"]
    policy = policy_cls()
    env_args = OmegaConf.to_container(cfg.env, resolve=True)
    env_args = {k: v for k, v in env_args.items() if k not in ["name", "start", "end"]}
    env_args["obs_keys"] = policy.obs_keys
    env_args["interval"] = "1m"

    # BNH features func
    ft_name = "bnh@features"
    feat_func = FUNCTION_REGISTRY[ft_name]
    col_range_dict = {}

    print(f"\n[screen_bt] BNH backtest results:")
    for symbol, df in minute_dfs.items():
        df = df.dropna(subset=["open", "close", "high", "low"]).copy()
        if len(df) == 0:
            print(f"  {symbol:10s}  SKIP (all NaN)")
            continue
        feat_df, col_range_dict = feat_func(df)
        for col in feat_df.columns:
            df[col] = feat_df[col]
        env = PaperTrade(df=df, **{**env_args, "col_range_dict": col_range_dict.copy()})
        request = BTRequest(policy_hyper_param={}, df=df, param_id=0, df_name=symbol)
        result = bt_rollout(request, policy, env)
        print(f"  {symbol:10s}  pnl={result.pnl*100:+.2f}%  bars={result.total_t}  pos_chg={result.pos_chg}")


def vis_mode(cfg: DictConfig, dfs: dict[str, pd.DataFrame], col_range_dict: dict) -> None:
    
    chart = Chart(toolbox=True, inner_width=1, inner_height=cfg.gui.mainchart_height)
    chart.candle_style(down_color="#00ff55", up_color="#ed4807")
    for df_name, df in dfs.items():
        policy_cls = POLICY_REGISTRY[cfg.policy.name]
        policy_args = {k: v for k, v in cfg.policy.items() if k not in ["name"]}
        policy = policy_cls(**policy_args)
        env_args = OmegaConf.to_container(cfg.env, resolve=True)  # to dict
        env_args = {k: v for k, v in env_args.items() if k not in ["name", "start", "end"]}
        env_args["obs_keys"] = policy.obs_keys
        env_args["interval"] = cfg.data.interval
        env_args["col_range_dict"] = col_range_dict
        env = PaperTrade(df =df, **env_args)
        obs = env.reset()
        done = False
        labels = []
        entry_dates = []
        exit_dates = []
        param  = policy.randomize_hyper_param(random_type=cfg.mode.hyper_search)
        policy.set_hyper_param(param)
        policy.init_policy()
        while not done:
            action, action_info = policy(obs)
            obs, reward, done, info = env.step(action)
            if action_info["entry_point"]:
                labels.append(("B", env.df.index[env._t]))
                entry_dates.append(env.df.index[env._t])
            if action_info["exit_point"]:
                labels.append(("S", env.df.index[env._t]))
                exit_dates.append(env.df.index[env._t])
        
        vis_lightweight_chart_df(
            chart=chart,
            df=df,
            df_name=df_name,
            entry_dates=entry_dates,
            exit_dates=exit_dates,
            labels=labels,
            mainchart_keys=cfg.gui.mainchart_keys,
            subchart_keys=cfg.gui.subchart_keys,
            mainchart_height=cfg.gui.mainchart_height,
        )


@hydra.main(config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    if cfg.mode.name == "screen_bt":
        screen_bt_mode(cfg)
        return None

    dfs = load_data(cfg)
    dfs, col_range_dict = make_ta_features(cfg, dfs)

    if cfg.mode.name == "vis":
        vis_mode(cfg, dfs, col_range_dict)
    elif cfg.mode.name == "bt":
        bt_mode(cfg, dfs, col_range_dict)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode.name}")
    return None


if __name__ == "__main__":
    main()
