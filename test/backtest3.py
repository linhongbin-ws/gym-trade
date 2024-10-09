from gym_trade.api import make_env, get_args, screen_daily
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gym_trade.tool.parallel import Parallel

from dataclasses import dataclass
env, env_config = make_env(tags=[], seed=0)
hdf = "~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5"
stock_dates = screen_daily(hdf, [("new_high", {"period": 52}), ("price_limit", {"upper_limit": 100, "lower_limit": 5}) ], return_high=True)
# print(stock_dates)

root_minute_dir = "~/ssd/data/stock-data/us-minute/"
csv_list = []
for k,v in tqdm(stock_dates.items()):
    for i in range(len(v[0])):
        ts = pd.to_datetime(str(v[0][i])) 
        date = ts.strftime('%Y-%m-%d')
        minute_path = Path(root_minute_dir).expanduser()
        minute_path = minute_path / ("kminute-"+str(date)) / (str(date+"-"+k+".csv"))
        # print(minute_path)
        if minute_path.exists():
            csv_list.append([str(minute_path),v[1][i]])
        # print(str(minute_path))
print("csv_list: ", len(csv_list))
# print(csv_list)

def backtest(csv_file,high_price):
    env, env_config = make_env(tags=[], seed=0)
    env.load_stock_list([csv_file])
    obs = env.reset()
    done = False
    while not done:
        if obs['high']>high_price:
            action = 0 
        elif obs['timestep']>389:
            action =1
        else:
            action = 2
        obs, reward, done, info = env.step(action)
    return env.pnl

# @dataclass
# class csv_request(BaseRequest):
#     csv: str

class parallel(Parallel):
    def parallel_func(self, input):
        csv_file = input[0]
        high_price = input[1]
        pnl = backtest(csv_file, high_price)
        file_name = Path(csv_file).stem
        return [pnl, file_name]
        # print(csv_file)


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
