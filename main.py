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
        _dfs.append(df)
    
    return _dfs, col_range_dict
    

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
    args = {k:v for k, v in cfg.policy.items() if k not in ["name"]}
    policy = policy_cls(**args)

    # create env
    args = OmegaConf.to_container(cfg.env, resolve=True) # to dict
    args = {k:v for k,v in args.items() if k not in ['name', 'start', 'end']}
    args["obs_keys"] = policy.obs_keys 
    args["interval"] = cfg.data.interval 
    for df in _df_list:
        env = PaperTrade(df_list=[df], col_range_dict= col_range_dict, **args)
    
    policy.observation_space = env.observation_space
    
    search_num = 1 if cfg.mode.hyper_search_type is None else cfg.mode.hyper_search_num
    best_pnl_stat = None
    for i in tqdm(range(search_num)):
        policy.init_policy(None if i == 0 else cfg.mode.hyper_search_type)
        obs = env.reset()
        done = False
        pos_prv = obs["dash@pos"]
        pos_chg = 0
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            if obs["dash@pos"] != pos_prv:
                pos_chg += 1
            if env._t % 500 == 0:
                print(f"action {action}, reward {reward}, progress {env._t}/{len(env.df.index)-1} ", end='\r')
            pos_prv = obs["dash@pos"]
        
        if best_pnl_stat is None:
            best_pnl_stat = {}
            best_pnl_stat["pnl"] = env.pnl
            best_pnl_stat["hyper_param"] = policy.hyper_param
        else:
            if env.pnl > best_pnl_stat["pnl"]:
                best_pnl_stat = {}
                best_pnl_stat["pnl"] = env.pnl
                best_pnl_stat["hyper_param"] = policy.hyper_param
        print(f"pnl: {env.pnl}, best pnl: {best_pnl_stat['pnl']}, position change: {pos_chg} / {len(env.df.index)-1} ")
    # mainchart_keys = [k for k in env.df.columns if k.startswith(tuple(cfg.gui.mainchart_types)) ]
    # subchart_keys = [k for k in env.df.columns if k.startswith(tuple(cfg.gui.subchart_types)) ] 
    if not cfg.general.no_vis:
        vis_lightweight_chart_df(env.df, mainchart_keys=cfg.policy.mainchart_keys , 
                                    subchart_keys=cfg.policy.subchart_keys , 
                                    mainchart_height=cfg.gui.mainchart_height)
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
    chart.set(df)
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

 