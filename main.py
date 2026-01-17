import os
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from lightweight_charts import Chart
import numpy as np
import pandas as pd
from gym_trade.tool.hydra import merge_ta_groups,DictConfig, OmegaConf
import hydra
import random
from gym_trade.tool.ta import make_ta
from gym_trade.env.embodied.gym_trade import GymTradeEnv
import yfinance as yf
from datetime import datetime
from pathlib import Path 
from gym_trade.policy.registry import POLICY_REGISTRY

def load_data(cfg: DictConfig) -> list[pd.DataFrame]:
    # print(cfg)
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
            cache_csv_dir =Path(cfg.data.cache_dir) /cfg.data.interval /f"{cfg.data.symbol}.csv" 
            if cfg.data.use_cache and cache_csv_dir.exists(): 
                df = pd.read_csv(cache_csv_dir)
            else:
                df = yf.download(cfg.data.symbol, multi_level_index=False, **args)
                if not cache_csv_dir.parent.exists():
                    cache_csv_dir.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_csv_dir)
                
            df = standardlize_df(df) 
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
    # print(cfg.ta.keys())

    # make feature
    _dfs = []
    for df in dfs:
        unfinish_dict = {k: v for k,v in cfg.ta.items()}
        ta_len_prv = len(unfinish_dict.keys()) + 1
        while not ta_len_prv==len(unfinish_dict.keys()):
            df, unfinish_dict = make_ta(df, cfg.ta)
            ta_len_prv = len(unfinish_dict.keys())
        assert len(unfinish_dict.keys()) == 0, f"unfinish ta {unfinish_dict.keys()} "
        _dfs.append(df)
    
    return _dfs
    

def bt(cfg: DictConfig, df_list: list[pd.DataFrame]) -> None:
    _df_list = []
    for df in df_list: 
        if cfg.mode.start is not None:
            date = datetime.strptime(cfg.mode.start, "%Y-%m-%d")
            df = df.truncate(after=date)
        if cfg.mode.end is not None:
            date = datetime.strptime(cfg.mode.end, "%Y-%m-%d")
            df = df.truncate(before=date)
        _df_list.append(df)
    args = OmegaConf.to_container(cfg.mode, resolve=True)
    args = {k:v for k,v in args.items() if k not in ['name', 'start', 'end']}

    for df in _df_list:
        env = GymTradeEnv(task="us_stock", df_list=[df],**args)

    print(f"avaliable poliy", print(POLICY_REGISTRY.keys()))
    policy_cls = POLICY_REGISTRY[cfg.policy.name]
    args = {k:v for k, v in cfg.policy.items() if k not in ["name"]}
    policy = policy_cls(**args)
    
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        print(f"action {action}, reward {reward}")
    print(f"pnl: {env.pnl}")
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





def vis(cfg: DictConfig, df: pd.DataFrame) -> None:
    # # download data
    # if cfg.network.use_cache:
    #     df = pd.read_csv(os.path.join(cfg.network.cache_dir, 'dow.csv'))
    # else:
    #     df = yf.download(['^DJI'], period='max',multi_level_index=False)
    #     if not os.path.exists(cfg.network.cache_dir):
    #         os.makedirs(cfg.network.cache_dir)
    #     df.to_csv(os.path.join(cfg.network.cache_dir, 'dow.csv'))

    df = standardlize_df(df)
    main_chart_width = 0.6
    chart = Chart(toolbox=True,inner_width=1,inner_height=main_chart_width)
    chart.candle_style(down_color='#00ff55', up_color='#ed4807')
    chart.set(df)
    random_color = lambda : f'rgba({random.randint(100, 255)}, {random.randint(100, 255)}, {random.randint(100, 255)}, 0.9)'

    for k in cfg.mode.ta_select:
        if k not in cfg.ta:
            continue
        v = cfg.ta[k]
        _cfgs = v.copy()
        _cfgs = OmegaConf.to_container(_cfgs, resolve=True)
        func = _cfgs['func']
        _cfgs.pop('func')
        call = getattr(ta_tool, func)
        results = call(df,**_cfgs)
        if isinstance(results, pd.DataFrame):
            for col in results.columns:
                df[k+'@'+col] = results[col]
        elif isinstance(results, pd.Series):
            df[k] = results
        else:
            raise NotImplementedError(f"Unsupported return type: {type(results)}")
    if cfg.mode.mainchart is not None:
        for k in cfg.mode.mainchart:
            line_df = pd.DataFrame({
                'time': df.index,
                k: df[k]
            })
            line_df = line_df.dropna()
            line = chart.create_line(k, color = random_color(),)
            chart.legend(True)
            line.set(line_df)
    if cfg.mode.subchart is not None:
        for k in cfg.mode.subchart:
            subchart = chart.create_subchart(position='left', width=1, height=(1-main_chart_width)/len(cfg.subchart),sync=True)
            subchart.legend(True)
            line = subchart.create_line(k)
            line_df = pd.DataFrame({
                'time': df.index,
                k: df[k]
            })
            # line_df = line_df.dropna()
            line.set(line_df)
        


    chart.show(block=True)


@hydra.main(config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:


    dfs = load_data(cfg)
    dfs = make_ta_features(cfg, dfs)
    if cfg.mode.name == 'vis':
        vis(cfg, dfs)
    elif cfg.mode.name == 'bt':
        bt(cfg, dfs)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode.mode}")
    return None

if __name__ ==  '__main__':
    main()

 