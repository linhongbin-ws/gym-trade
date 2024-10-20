from gym_trade.api import make_env, get_args, screen_daily
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gym_trade.tool.parallel import Parallel

from dataclasses import dataclass
env, env_config = make_env(tags=[], seed=0)
hdf = "~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5"
stock_dates = screen_daily(hdf, [("new_high", {"period": 52}), 
                                 ("price_limit", {"upper_limit": 100, "lower_limit": 5}) ], 
                           symbol_list=["GME"])
print(stock_dates)

root_minute_dir = "~/ssd/data/stock-data/us-minute/"
csv_list = []
is_break = False
for k,v in tqdm(stock_dates.items()):
    for i in range(len(v[0])):
        ts = pd.to_datetime(str(v[0][i])) 
        date = ts.strftime('%Y-%m-%d')
        minute_path = Path(root_minute_dir).expanduser()
        minute_path = minute_path / ("kminute-"+str(date)) / (str(date+"-"+k+".csv"))
        # print(minute_path)
        if minute_path.exists():
            csv_list.append([str(minute_path),v[1][i], v[2][i]])

        # print(str(minute_path))
print("csv_list: ", len(csv_list))
# print(csv_list)

def backtest(csv_file,high_price, close_price):
    env, env_config = make_env(tags=[], seed=0)
    env = LightChart_Visualizer(env)
    env.load_stock_list([csv_file])
    file_name = Path(csv_file).stem
    obs = env.reset()
    calibrate_high_price = high_price * env.unwrapped.df['close'].iloc[-1] / close_price
    env.gui_init()
    env.gui_horizon_line(calibrate_high_price, text="previous high")
    env.gui_textbox("symbol", file_name)
    done = False
    new_high = obs['high']
    break_keep_count = 0
    max_break_keep = 2
    bk_or_prv = obs['break_high_or']
    print(obs.keys())
    while not done:
        if obs['break_high'] and obs["position_ratio"]<=0.1 and env.timestep > 0 and obs['high'] > calibrate_high_price:
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
        # print(env.unwrapped.timestep, action)

        if action==0:
            env.gui_marker("buy")
            print(f"buy at timestep {env.timestep}")
        if action==1:
            env.gui_marker("sell")
            print(f"sell at timestep {env.timestep}")
    env.gui_textbox("pnl", "pnl: " + str(env.pnl))
    env.gui_show()
    return env.pnl

for csv_file in csv_list: 
    pnl = backtest(csv_file[0],csv_file[1], csv_file[2] )
# pnl_results = []
# print(csv_list)
# for csv in tqdm(csv_list):
#     pnl = backtest(csv)
#     file_name = Path(csv).stem
#     pnl_results.append([pnl, file_name])
# print(pnl_results)
    

# p = parallel(worker_type="process")
# pnl_results = p.run(csv_list)
# df = pd.DataFrame(pnl_results,columns =['pnl', 'file'])
# print(df)
# print("mean pnl (%):",df['pnl'].mean())
# df.to_csv("./test/pnl.csv")
# p.close()
