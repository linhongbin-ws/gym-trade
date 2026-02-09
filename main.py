
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
from lightweight_charts import Chart
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from gym_trade.tool.ta import make_ta_all_safe, make_ta_all
from gym_trade.env.embodied import PaperTrade
from datetime import datetime
from pathlib import Path
from gym_trade.policy.registry import POLICY_REGISTRY
from tqdm import tqdm
import yaml
from queue import Empty as QEmpty
import multiprocessing as mp
from typing import Any

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


def load_data(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    dfs = load_data_func(
        data_api=cfg.data.name,
        proxy=cfg.general.proxy,
        interval=cfg.data.interval,
        start=cfg.data.start,
        end=cfg.data.end,
        symbols=cfg.data.symbol,
        cache_dir=cfg.data.cache_dir,
        force_download=cfg.data.force_download,
    )
    return dfs


def make_ta_features(cfg: DictConfig, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

    # merge cfg ta_xxx to ta
    OmegaConf.set_struct(cfg, False)
    keys = [k for k in cfg.keys()]
    cfg.ta = {}
    for k in keys:
        if k.startswith("ta_"):
            cfg.ta.update(cfg[k])
            del cfg[k]
    OmegaConf.set_struct(cfg, True)

    ta_dict = {k: v for k, v in cfg.ta.items() if k in cfg.policy.ta_select_keys}
    if cfg.general.future_check:
        _dfs, col_range_dict = make_ta_all_safe(dfs, ta_dict)
    else:
        _dfs, col_range_dict = make_ta_all(dfs, ta_dict)
    return _dfs, col_range_dict


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
    df: pd.DataFrame,
    df_name: str,
    entry_dates: list[pd.DatetimeIndex],
    exit_dates: list[pd.DatetimeIndex],
    mainchart_keys: list[str] = [],
    subchart_keys: list[str] = [],
    mainchart_height: float = 0.6,
):
    chart = Chart(toolbox=True, inner_width=1, inner_height=mainchart_height)
    chart.candle_style(down_color="#00ff55", up_color="#ed4807")
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
    
    for entry_date in entry_dates:
        chart.marker(text=f"B: {entry_date}", time=entry_date)
    for exit_date in exit_dates:
        chart.marker(text=f"S: {exit_date}", time=exit_date)

    chart.show(block=True)


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


def vis_mode(cfg: DictConfig, dfs: dict[str, pd.DataFrame], col_range_dict: dict) -> None:
    
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
        entry_dates = []
        exit_dates = []
        while not done:
            action, action_info = policy(obs)
            obs, reward, done, info = env.step(action)
            if action_info["entry_point"]:
                entry_dates.append(env._t)
            if action_info["exit_point"]:
                exit_dates.append(env._t)

        vis_lightweight_chart_df(
            df=df,
            df_name=df_name,
            entry_dates=entry_dates,
            exit_dates=exit_dates,
            mainchart_keys=cfg.gui.mainchart_keys,
            subchart_keys=cfg.gui.subchart_keys,
            mainchart_height=cfg.gui.mainchart_height,
        )


@hydra.main(config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    dfs = load_data(cfg)
    dfs, col_range_dict = make_ta_features(cfg, dfs)
    if cfg.mode.name == "vis":
        vis_mode(cfg, dfs, col_range_dict)
    elif cfg.mode.name == "bt":
        bt_mode(cfg, dfs, col_range_dict)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode.mode}")
    return None


if __name__ == "__main__":
    main()
