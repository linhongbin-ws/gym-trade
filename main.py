import os
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from lightweight_charts import Chart
import numpy as np
import pandas as pd
from gym_trade.tool.hydra import merge_ta_groups,DictConfig, OmegaConf
import hydra
import random
from gym_trade.tool import ta as ta_tool
from gym_trade.env.embodied.gym_trade import GymTradeEnv
import yfinance as yf
from datetime import datetime


def load_data(cfg: DictConfig):
    print(cfg)
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
            

        df = yf.download(cfg.data.symbol, multi_level_index=False, **args)
    else:
        raise NotImplementedError(f"Unsupported data source: {cfg.data_name}")
    return df

def bt(cfg: DictConfig, df: pd.DataFrame) -> None:
    if cfg.mode.start is not None:
        if cfg.data.interval == "1d":
            date = datetime.strptime(cfg.mode.start, "%Y-%m-%d")
        df = df.truncate(after=date)
    if cfg.mode.end is not None:
        if cfg.data.interval == "1d":
            date = datetime.strptime(cfg.mode.end, "%Y-%m-%d")
        df = df.truncate(before=date)
    df_list = [df]
    args = OmegaConf.to_container(cfg.mode, resolve=True)
    args = {k:v for k,v in args.items() if k not in ['name', 'start', 'end']}
    env = GymTradeEnv(task="us_stock",**args)

    

    #ta
    ta_dict_list = []
    ta_dict_list.append({"func":"direction_toggle", "key":"close"})
    env = TA(env, ta_dict_list=ta_dict_list)


    
    obs = env.reset()

    done = False
    sig_cnt = 0
    sig_cnt_thres =1
    while not done:
        if obs['direction_toggle_pattern_strongup_acc@close'] >=1:
            sig_cnt+=1
        else:
            sig_cnt=0

        if obs['position_ratio'] <0.9 and sig_cnt>=sig_cnt_thres:
            action = 1
        elif obs['position_ratio'] >0.9 and sig_cnt<sig_cnt_thres:
            action = -1
        else:
            action = -0

        if action>=0.1:
            env.gui_marker("buy")
            # print(f"buy at timestep {env.timestep}")
        if action<=-0.1:
            env.gui_marker("sell")
            print(f"sell at timestep {env.timestep}")
        obs, reward, done, info = env.step(action)
        print(f"time: {env.timestep}/ {len(env.df.index)-1}. reward: {reward}, pnl: {env.pnl}")





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


    df = load_data(cfg)

    if cfg.mode.mode == 'vis':
        vis(cfg, df)
    elif cfg.mode.mode == 'bt':
        bt(cfg, df)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode.mode}")
    return None

if __name__ ==  '__main__':
    main()

 