from gym_trade.api import make_env, get_args, screen_daily
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd


env, env_config = make_env(tags=[], seed=0)
hdf = "~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5"
stock_dates = screen_daily(hdf, [("new_high", {"period": 52})], return_high=True)
print(stock_dates)

root_minute_dir = "~/ssd/data/stock-data/us-minute/"
csv_list = []
for k,v in stock_dates.items():
    for d in v:
        ts = pd.to_datetime(str(d[0])) 
        date = ts.strftime('%Y-%m-%d')
        minute_path = Path(root_minute_dir).expanduser()
        minute_path = minute_path / ("kminute-"+str(date)) / (str(date+"-"+k+".csv"))
        # print(minute_path)
        if minute_path.exists():
            csv_list.append([str(minute_path),d[1]])
        # print(str(minute_path))


def backtest(csv_file):
    env, env_config = make_env(tags=[], seed=0)
    env.load_stock_list([csv_file])
    obs = env.reset()
    done = False
    while not done:
        if env.timestep == 1:
            action = 0 
        else:
            action = 2
        obs, reward, done, info = env.step(action)
    return env.pnl


pnl_results = []
print(csv_list)
for csv in tqdm(csv_list):
    pnl = backtest(csv)
    file_name = Path(csv).stem
    pnl_results.append([pnl, file_name])
print(pnl_results)
df = pd.DataFrame(pnl_results)
df.to_csv("./test/pnl.csv")
