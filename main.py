from dataclasses import dataclass, asdict
import os
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from lightweight_charts import Chart
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from gym_trade.tool.ta import make_ta
from gym_trade.env.embodied import PaperTrade
import yfinance as yf
from datetime import datetime
from pathlib import Path 
from gym_trade.policy.registry import POLICY_REGISTRY
from tqdm import tqdm
import yaml
from queue import Empty as QEmpty
import uuid
import multiprocessing as mp
from typing import Any



def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    else:
        return obj

def load_data(cfg: DictConfig) -> list[pd.DataFrame]:
    if cfg.data.name == 'yfinance':
        if cfg.general.proxy is not None:
            os.environ['HTTP_PROXY'] = cfg.general.proxy
            os.environ['HTTPS_PROXY'] = cfg.general.proxy
            print(f"set proxy to {cfg.general.proxy}")
        args = {"interval": cfg.data.interval,
         "period":'max'}
        if cfg.data.start is not None:
            args['start'] = cfg.data.start
        if cfg.data.end is not None:
            args['end'] = cfg.data.end
        
        dfs = []
        for symbol in cfg.data.symbol:
            cache_csv_dir =Path(cfg.data.cache_dir) /cfg.data.interval /f"{symbol}.csv" 
            if cfg.data.use_cache and cache_csv_dir.exists(): 
                df = pd.read_csv(cache_csv_dir)
            else:
                df = yf.download(symbol, multi_level_index=False, **args)
                if not cache_csv_dir.parent.exists():
                    cache_csv_dir.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_csv_dir)
                
            df = standardlize_df(df) 
            if cfg.data.interval == "1m": 
                df = fill_missing_frame(df)# filling missing frame
            dfs.append(df)
    else:
        raise NotImplementedError(f"Unsupported data source: {cfg.data_name}")
    return dfs

def make_ta_features(cfg: DictConfig, dfs: list[pd.DataFrame]) -> pd.DataFrame: 

    # merge cfg ta_xxx to ta
    OmegaConf.set_struct(cfg, False)
    keys = [k for k in cfg.keys()]
    cfg.ta = {}
    for k in keys:
        if k.startswith("ta_"):
            cfg.ta.update(cfg[k])
            del cfg[k]
    OmegaConf.set_struct(cfg, True)


    # make feature
    _dfs = []
    for df in dfs:
        unfinish_dict = {k: v for k,v in cfg.ta.items() if k in cfg.policy.ta_select_keys}
        ta_len_prv = len(unfinish_dict.keys()) + 1
        col_range_dict = None
        while not ta_len_prv==len(unfinish_dict.keys()):
            df, col_range_dict, unfinish_dict = make_ta(df, cfg.ta, col_range_dict=col_range_dict)
            ta_len_prv = len(unfinish_dict.keys())
        assert len(unfinish_dict.keys()) == 0, f"unfinish ta {unfinish_dict.keys()} "
        df['index_datetime'] = df.index.values
        _dfs.append(df)
    
    return _dfs, col_range_dict

@dataclass
class BTRequest:
    policy_hyper_param: dict
    id: str

@dataclass
class BTResult:
    pnl: float
    policy_hyper_param: dict
    pos_chg: int
    id: str
    hold_t: int 
    total_t: int 


def bt_rollout(request: BTRequest, policy, env: PaperTrade, stop_event = None):
    policy.set_hyper_param(request.policy_hyper_param)
    policy.init_policy()
    obs = env.reset()
    done = False
    pos_prv = obs["dash@pos"]
    pos_chg = 0
    hold_cnt = 0
    total_t = 0
    while not done:
        if stop_event is not None:
            if stop_event.is_set():
                break
        total_t += 1
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if obs["dash@pos"] != pos_prv:
            pos_chg += 1
        if obs["dash@pos"] > 0:
            hold_cnt += 1
        # if env._t % 500 == 0:
            # print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
        pos_prv = obs["dash@pos"]
    
    result = BTResult(pnl=env.pnl, policy_hyper_param=policy.hyper_param, pos_chg=pos_chg, id=request.id, hold_t=hold_cnt, total_t=total_t)
    return result


def bt_server_loop(policy_name,policy_args, env_args, df_list, stop_event, request_queue, result_queue):
    policy_cls = POLICY_REGISTRY[policy_name]
    policy = policy_cls(**policy_args)
    env = PaperTrade(df_list=df_list, **env_args)
    
    while not stop_event.is_set():
        try:
            request = request_queue.get(timeout=0.5)  # 定期醒来检查 stop_event
        except QEmpty:
            continue

        
        # policy.set_hyper_param(request.policy_hyper_param)
        # policy.init_policy()
        # obs = env.reset()
        # done = False
        # pos_prv = obs["dash@pos"]
        # pos_chg = 0
        # hold_cnt = 0
        # total_t = 0
        # while not done and not stop_event.is_set():
        #     total_t += 1
        #     action = policy(obs)
        #     obs, reward, done, info = env.step(action)
        #     if obs["dash@pos"] != pos_prv:
        #         pos_chg += 1
        #     if obs["dash@pos"] > 0:
        #         hold_cnt += 1
        #     # if env._t % 500 == 0:
        #         # print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
        #     pos_prv = obs["dash@pos"]
        result = bt_rollout(request, policy, env, stop_event)
        # result = BTResult(pnl=env.pnl, policy_hyper_param=policy.hyper_param, pos_chg=pos_chg, id=request.id, hold_t=hold_cnt, total_t=total_t)
        result_queue.put(result)


class BTServer:
    """recieve a bt request, and return a result"""
    def __init__(self,
            cfg: DictConfig,
            policy_name: str,
            policy_args: dict[str, Any],
            env_args: dict[str, Any],
            df_list: list[pd.DataFrame],
            n_workers: int = 2,
    ):
        self._n_workers = cfg.mode.workers 

        if self._n_workers > 1:
            ctx = mp.get_context("spawn")  # 跨平台更稳（Windows/macOS 必须 spawn）
            self._request_queue = ctx.Queue(maxsize=self.n_workers * 4)
            self._result_queue= ctx.Queue()
            self._stop_event = ctx.Event()


            self.procs = [
                ctx.Process(target=bt_server_loop, args=(policy_name,policy_args, env_args, df_list, self._stop_event, self._request_queue , self._result_queue))
                for _ in range(self.n_workers)
            ]
            for p in self.procs:
                # p.daemon = True
                p.start()
            
        self._closed = False
        self._cfg = cfg
        self._policy_args = policy_args
        self._env_args = env_args
        self._df_list = df_list
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

    def backtest(self):
 
        
        policy_cls = POLICY_REGISTRY[self._cfg.policy.name]
        policy = policy_cls(**self._policy_args)
        best_pnl_stat = None
        file_name = "bt_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ ".yaml"

        def deal_with_best_pnl(best_pnl_stat, new_result):
            if best_pnl_stat is None:
                best_pnl_stat = asdict(result)
            elif result.pnl > best_pnl_stat["pnl"]:
                best_pnl_stat = asdict(result)
            else:
                return best_pnl_stat
            
            # print(f"best pnl: {best_pnl_stat['pnl']}, position change: {result.pos_chg} ")
            # print(f"pnl: {env.pnl}, best pnl: {best_pnl_stat['pnl']}, position change: {pos_chg} / {len(env.df.index)-1} ")
            pbar.write(f"best pnl: {best_pnl_stat['pnl']:.3f}, pos chg: {result.pos_chg} , hold t: {result.hold_t} / {result.total_t} ")
            save_result_dir = Path(self._cfg.mode.save_result_dir)
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


        with tqdm(total=self._cfg.mode.search_num) as pbar:

            if self._n_workers == 1:
                policy_cls = POLICY_REGISTRY[self._policy_name]
                policy = policy_cls(**self._policy_args)
                env = PaperTrade(df_list=self._df_list, **self._env_args)
                for i in range(self._cfg.mode.search_num):
                    policy_hyper_param = policy.randomize_hyper_param(random_type=self._cfg.mode.hyper_search)
                    request = BTRequest(policy_hyper_param=policy_hyper_param, id=str(uuid.uuid4()))
                    result = bt_rollout(request, policy, env, stop_event=None)
                    pbar.update(1)
                    best_pnl_stat = deal_with_best_pnl(best_pnl_stat, result)
            else:
                while not self._stop_event.is_set() and pbar.n <= self._cfg.mode.search_num:
                    if not self._request_queue.full():
                        policy_hyper_param = policy.randomize_hyper_param(random_type=self._cfg.mode.hyper_search)
                        request = BTRequest(policy_hyper_param=policy_hyper_param, id=str(uuid.uuid4()))
                        self._request_queue.put(request)

                    if not self._result_queue.empty():
                        try:
                            result = self._result_queue.get(timeout=0.5)  # 定期醒来检查 stop_event
                            pbar.update(1)
                        except QEmpty:
                            continue
                        best_pnl_stat = deal_with_best_pnl(best_pnl_stat,result)
                        # if result.pnl > self._best_pnl:
                        #     self._best_pnl = result.pnl
                        #     self._best_hyper_param = result.hyper_param
                        
                        # if best_pnl_stat is None:
                        #     best_pnl_stat = asdict(result)
                        # elif result.pnl > best_pnl_stat["pnl"]:
                        #     best_pnl_stat = asdict(result)
                        # else:
                        #     continue
                        
                        # # print(f"best pnl: {best_pnl_stat['pnl']}, position change: {result.pos_chg} ")
                        # # print(f"pnl: {env.pnl}, best pnl: {best_pnl_stat['pnl']}, position change: {pos_chg} / {len(env.df.index)-1} ")
                        # pbar.write(f"best pnl: {best_pnl_stat['pnl']:.3f}, pos chg: {result.pos_chg} , hold t: {result.hold_t} / {result.total_t} ")
                        # save_result_dir = Path(self._cfg.mode.save_result_dir)
                        # save_result_dir.mkdir(parents=True, exist_ok=True)
                        # file = save_result_dir / file_name
                        # best_pnl_stat_save = {"best_pnl": best_pnl_stat}
                        # with open(file, "w", encoding="utf-8") as f:
                        #     yaml.dump(
                        #         to_python(best_pnl_stat_save),
                        #         f,
                        #         default_flow_style=False,
                        #         allow_unicode=True,
                        #         sort_keys=False,
                        #         indent=4,
                        #     )

        
            # # print(None if i == 0 else cfg.mode.hyper_search_type)
            # # assert False, None if i == 0 else cfg.mode.hyper_search_type
            # policy.init_policy(None if i == 0 else cfg.mode.hyper_search_type)
            # obs = env.reset()
            # done = False
            # pos_prv = obs["dash@pos"]
            # pos_chg = 0
            # while not done:
            #     action = policy(obs)
            #     obs, reward, done, info = env.step(action)
            #     if obs["dash@pos"] != pos_prv:
            #         pos_chg += 1
            #     if env._t % 500 == 0:
            #         print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
            #     pos_prv = obs["dash@pos"]
            
            


def bt(cfg: DictConfig, df_list: list[pd.DataFrame], col_range_dict: dict) -> None:
    _df_list = []
    for df in df_list: 
        if cfg.mode.start is not None:
            date = datetime.strptime(cfg.mode.start, "%Y-%m-%d")
            if cfg.data.interval == "1m": 
                date = date.replace(hour=9, minute=30)

            df = df.truncate(before=date)
        if cfg.mode.end is not None:
            date = datetime.strptime(cfg.mode.end, "%Y-%m-%d")
            if cfg.data.interval == "1m": 
                date = date.replace(hour=4, minute=00)
            df = df.truncate(after=date)
        _df_list.append(df)


    # create policy
    print(f"avaliable poliy", print(POLICY_REGISTRY.keys()))
    policy_cls = POLICY_REGISTRY[cfg.policy.name]
    policy_args = {k:v for k, v in cfg.policy.items() if k not in ["name"]}
    policy = policy_cls(**policy_args)

    # create env
    env_args = OmegaConf.to_container(cfg.env, resolve=True) # to dict
    env_args = {k:v for k,v in env_args.items() if k not in ['name', 'start', 'end']}
    env_args["obs_keys"] = policy.obs_keys 
    env_args["interval"] = cfg.data.interval 
    env_args["col_range_dict"] = col_range_dict

    

    server = BTServer(cfg=cfg, 
            policy_name=cfg.policy.name, policy_args=policy_args, env_args=env_args, df_list=df_list,)
    server.backtest()
    server.shutdown()

    # policy.observation_space = env.observation_space
    
    # search_num = 1 if cfg.mode.hyper_search_type is None else cfg.mode.hyper_search_num
    # best_pnl_stat = None
    # file_name = "bt_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ ".yaml"
    # for i in tqdm(range(search_num)):
    #     # print(None if i == 0 else cfg.mode.hyper_search_type)
    #     # assert False, None if i == 0 else cfg.mode.hyper_search_type
    #     policy.init_policy(None if i == 0 else cfg.mode.hyper_search_type)
    #     obs = env.reset()
    #     done = False
    #     pos_prv = obs["dash@pos"]
    #     pos_chg = 0
    #     while not done:
    #         action = policy(obs)
    #         obs, reward, done, info = env.step(action)
    #         if obs["dash@pos"] != pos_prv:
    #             pos_chg += 1
    #         if env._t % 500 == 0:
    #             print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
    #         pos_prv = obs["dash@pos"]
        
    #     if best_pnl_stat is None:
    #         best_pnl_stat = {}
    #         best_pnl_stat["pnl"] = env.pnl
    #         best_pnl_stat["hyper_param"] = policy.hyper_param
    #     else:
    #         if env.pnl > best_pnl_stat["pnl"]:
    #             best_pnl_stat = {}
    #             best_pnl_stat["pnl"] = env.pnl
    #             best_pnl_stat["hyper_param"] = policy.hyper_param
    #     print(f"pnl: {env.pnl}, best pnl: {best_pnl_stat['pnl']}, position change: {pos_chg} / {len(env.df.index)-1} ")
    
    #     save_result_dir = Path(cfg.mode.save_result_dir)
    #     save_result_dir.mkdir(parents=True, exist_ok=True)
    #     file = save_result_dir / file_name
    #     # print("save_result_dir =", save_result_dir)
    #     # print("file_name       =", file_name)

    #     best_pnl_stat_save = {"best_pnl": best_pnl_stat,
    #         "pos_chg": pos_chg}
    #     with open(file, "w", encoding="utf-8") as f:
    #         yaml.dump(
    #             to_python(best_pnl_stat_save),
    #             f,
    #             default_flow_style=False,
    #             allow_unicode=True,
    #             sort_keys=False,
    #             indent=4,
    #         )





    # mainchart_keys = [k for k in env.df.columns if k.startswith(tuple(cfg.gui.mainchart_types)) ]
    # subchart_keys = [k for k in env.df.columns if k.startswith(tuple(cfg.gui.subchart_types)) ] 
    # if not cfg.general.no_vis:
    #     vis_lightweight_chart_df(env.df, mainchart_keys=cfg.policy.mainchart_keys , 
    #                                 subchart_keys=cfg.policy.subchart_keys , 
    #                                 mainchart_height=cfg.gui.mainchart_height)
    # while not done:
    #     if obs['direction_toggle_pattern_strongup_acc@close'] >=1:
    #         sig_cnt+=1
    #     else:
    #         sig_cnt=0

    #     if obs['position_ratio'] <0.9 and sig_cnt>=sig_cnt_thres:
    #         action = 1
    #     elif obs['position_ratio'] >0.9 and sig_cnt<sig_cnt_thres:
    #         action = -1
    #     else:
    #         action = -0

    #     if action>=0.1:
    #         env.gui_marker("buy")
    #         # print(f"buy at timestep {env.timestep}")
    #     if action<=-0.1:
    #         env.gui_marker("sell")
    #         print(f"sell at timestep {env.timestep}")
    #     obs, reward, done, info = env.step(action)
    #     print(f"time: {env.timestep}/ {len(env.df.index)-1}. reward: {reward}, pnl: {env.pnl}")


def vis_lightweight_chart_df(df,
    mainchart_keys: list[str] = [],
    subchart_keys: list[str] = [],
    mainchart_height: float = 0.6, 
   ):
    chart = Chart(toolbox=True,inner_width=1,inner_height=mainchart_height)
    chart.candle_style(down_color='#00ff55', up_color='#ed4807')
    _df = df[['close', 'open', 'high', 'low', 'volume']]
    _df["time"] = df.index
    chart.set(_df)
    random_color = lambda : f'rgba({random.randint(100, 255)}, {random.randint(100, 255)}, {random.randint(100, 255)}, 0.9)'


    lines = {}

    # assert False, mainchart_keys
    for k in mainchart_keys:
        line_df = pd.DataFrame({
            'time': df.index,
            k: df[k]
        })
        # line_df = line_df.dropna()
        lines[k] = chart.create_line(k, color = random_color(),)
        chart.legend(True)
        lines[k].set(line_df)

    for k in subchart_keys:
        subchart = chart.create_subchart(position='left', width=1, height=(1-mainchart_height)/len(subchart_keys),sync=True)
        subchart.legend(True)
        lines[k] = subchart.create_line(k)
        line_df = pd.DataFrame({
            'time': df.index,
            k: df[k]
        })
        # line_df = line_df.dropna()
        lines[k].set(line_df)
        


    chart.show(block=True)



def vis(cfg: DictConfig, df_list: list[pd.DataFrame]) -> None:
    for df in df_list: 
        vis_lightweight_chart_df(df, mainchart_keys=cfg.gui.mainchart_keys, subchart_keys=cfg.gui.subchart_keys , mainchart_height=cfg.gui.mainchart_height) 
 


@hydra.main(config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:


    dfs = load_data(cfg)
    dfs, col_range_dict = make_ta_features(cfg, dfs)
    if cfg.mode.name == 'vis':
        vis(cfg, dfs)
    elif cfg.mode.name == 'bt':
        bt(cfg, dfs, col_range_dict)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode.mode}")
    return None

if __name__ ==  '__main__':
    main()

 