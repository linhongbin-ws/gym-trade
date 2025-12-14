from gym_trade.api import make_env, get_args, screen_daily
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gym_trade.tool.parallel import Parallel
import time
from copy import deepcopy

from dataclasses import dataclass
env, env_config = make_env(tags=[], seed=0)
hdf = "~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5"
is_parallel = False
gui = True
stock_dates = screen_daily(hdf, 
                           [
                                ("price_limit", {"upper_limit": 100, "lower_limit": 5}) , 
                                ("new_high", {"period": 52}), 
                                 ("new_volume_high", {"period": 52})
                                ],
                        symbol_list=["GME"],
                           )
cnt = 0
for k,v in stock_dates.items():
    for _ in range(len(v['dates'])):
        cnt+=1
print(f"screen stock dates num: {cnt}")


root_minute_dir = "~/ssd/data/stock-data/us-minute/"
csv_list = []
is_break = False
for k,v in tqdm(stock_dates.items()):
    for i in range(len(v['dates'])):
        ts = pd.to_datetime(str(v['dates'][i])) 
        date = ts.strftime('%Y-%m-%d')
        minute_path = Path(root_minute_dir).expanduser()
        minute_path = minute_path / ("kminute-"+str(date)) / (str(date+"-"+k+".csv"))
        # print(minute_path)
        if minute_path.exists():
            args = {nk: nv[i] for nk, nv in v.items()}
            args['minute_path'] = str(minute_path)
            csv_list.append(args)

        # print(str(minute_path))
print("csv_list: ", len(csv_list))
# print(csv_list)

def backtest(minute_path, Prv_High, Prv_Volme_High, Close, gui=False,timeout=10, **kwargs):
    if timeout>0:
        start_time = time.time()
    env, env_config = make_env(tags=[], seed=0)
    if gui:
        env = LightChart_Visualizer(env)
    env.load_stock_list([minute_path])
    file_name = Path(minute_path).stem
    obs = env.reset()
    calibrate_high_price = Prv_High * env.unwrapped.df['close'].iloc[-1] / Close
    calibrate_volume_high = Prv_Volme_High / (env.unwrapped.df['close'].iloc[-1] / Close)
    if gui:
        env.gui_init()
        env.gui_horizon_line(calibrate_high_price, text="previous high")
        env.gui_textbox("symbol", file_name)
    done = False
    new_high = obs['high']
    break_keep_count = 0
    max_break_keep = 2
    bk_or_prv = obs['break_high_or']
    # print(obs.keys())
    agg_volume = obs['volume']
    while not done:
        if timeout >0 and (time.time()- start_time > timeout):
            print("timeout..")
            return None

        if obs['break_high'] \
            and obs["position_ratio"]<=0.1 \
            and env.timestep > 0 \
            and obs['high'] > calibrate_high_price\
            and agg_volume > calibrate_volume_high\
                :
            action = 0 
        elif obs['timestep']>=388 and obs["position_ratio"]>0.1:
            action =1
            # print("sell when market close")
        elif (bk_or_prv and (not obs['break_high_or'])) and obs["position_ratio"]>0.1:
            action =1
            # print("sell")
        else:
            action = 2
        
        bk_or_prv = obs['break_high_or']
        obs, reward, done, info = env.step(action)
        agg_volume+=obs['volume']
        # print(env.unwrapped.timestep, action)
        if gui:
            if action==0:
                env.gui_marker("buy")
                print(f"buy at timestep {env.timestep}")
            if action==1:
                env.gui_marker("sell")
                print(f"sell at timestep {env.timestep}")
    if gui:
        env.gui_textbox("pnl", "pnl: " + str(env.pnl))
        env.gui_show()
    return env.pnl

if not is_parallel:
    pnl_results = []
    for args in tqdm(csv_list): 
        pnl = backtest(gui=gui,**args)
        file_name = Path(args['minute_path']).stem
        pnl_results.append([pnl, file_name])
else: 
    class parallel(Parallel):
        def parallel_func(self, input):
            pnl = backtest(**input)
            file_name = Path(input['minute_path']).stem
            return [pnl, file_name]
            # print(csv_file)
    p = parallel(worker_type="process")
    pnl_results = p.run(csv_list)
    pnl_results = [v for v in pnl_results if v is not None]
    
df = pd.DataFrame(pnl_results,columns =['pnl', 'file'])
print(df)
print("mean pnl (%):",df['pnl'].mean(),"std pnl (%):",df['pnl'].std(),)
df.to_csv("./test/pnl.csv")

if is_parallel:
    p.close()




