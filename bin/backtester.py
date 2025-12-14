from gym_trade.api import make_env, get_args, screen_daily, backtest
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gym_trade.tool.parallel import Parallel
import time
import argparse
from datetime import datetime
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--daydir', type=str, default="~/ssd/data/stock-data/us-daily/kdaily-2024-10-06/US-daily-20190101-20241006.h5")
parser.add_argument('--mindir', type=str, default="~/ssd/data/stock-data/us-minute/")
parser.add_argument('--worker', type=int, default=1)
parser.add_argument('--policy', type=str, default="breakout")
parser.add_argument('--symbol', type=str, nargs='+', default=[])
parser.add_argument('--savedir', type=str, default='./data/backtest')
parser.add_argument('--gui', action="store_true")
parser.add_argument('--date', type=str, nargs='+', default=[])
parser.add_argument('--csv', type=str, default="")
parser.add_argument('--csv-num', type=int, default=6)
in_args = parser.parse_args()

env, env_config = make_env(tags=in_args.env_tag, seed=0)
hdf = in_args.daydir
root_minute_dir = in_args.mindir
is_parallel = in_args.worker > 1
gui = in_args.gui


if in_args.csv =="":
    # screen
    stock_dates = screen_daily(hdf, 
                            env_config.screen,
                                symbol_list=in_args.symbol,
                                dates=in_args.date
                            )
else:
    _df = pd.read_csv(in_args.csv)
    _df.dropna(inplace=True)
    stock_dates = None
    if in_args.csv_num>0:
        _df = _df[:in_args.csv_num]
    elif in_args.csv_num<0:
        _df = _df.iloc[::-1]
        _df = _df[:-in_args.csv_num]
    for i in _df.index:
        file = _df['file'][i]
        _date = file[:10]
        _symbol = file[11:]
        sd = screen_daily(hdf, 
                        env_config.screen,
                            symbol_list=[_symbol],
                            dates=[_date]
                        )
        if stock_dates is None:
            stock_dates = sd
        else:
            for _k, _v in sd.items():
                if _k in stock_dates:
                    stock_dates[_k] = {_i: stock_dates[_k][_i]+sd[_k][_i] for _i, _j in _v}
                else:
                    stock_dates[_k] = sd[_k]
    

cnt = 0
for k,v in stock_dates.items():
    for d in range(len(v['dates'])):
        cnt+=1
print(f"screen num: {cnt}")



# check if minute files exit
csv_list = []
is_break = False
for k,v in tqdm(stock_dates.items()):
    for i in range(len(v['dates'])):
        ts = pd.to_datetime(str(v['dates'][i])) 
        date = ts.strftime('%Y-%m-%d')
        minute_path = Path(root_minute_dir).expanduser()
        minute_path = minute_path / ("kminute-"+str(date)) / (str(date+"-"+k+".csv"))
        if minute_path.exists():
            args = {nk: nv[i] for nk, nv in v.items()}
            args['minute_path'] = str(minute_path)
            csv_list.append(args)

print(f"screen exist / total: {len(csv_list)} / {cnt}")

# backtest
if not is_parallel:
    pnl_results = []
    for args in tqdm(csv_list): 
        new_args = deepcopy(args)
        new_args['policy'] = in_args.policy
        new_args['gui'] = gui
        new_args['env_tag'] = in_args.env_tag
        pnl = backtest(**new_args)
        file_name = Path(args['minute_path']).stem
        pnl_results.append([pnl, file_name])
else: 
    class parallel(Parallel):
        def parallel_func(self, input):
            global in_args
            new_args = deepcopy(input)
            new_args['policy'] = in_args.policy
            new_args['gui'] = False
            new_args['env_tag'] = in_args.env_tag
            pnl = backtest(**new_args)
            file_name = Path(input['minute_path']).stem
            return [pnl, file_name]
            # print(csv_file)
    p = parallel(worker_type="process", worker_num=in_args.worker)
    pnl_results = p.run(csv_list)
    pnl_results = [v for v in pnl_results if v is not None]

df = pd.DataFrame(pnl_results,columns =['pnl', 'file'])
df.dropna(inplace=True)
df = df[df['pnl'] != 0]
df.sort_values(by=['pnl'],inplace=True)
print(df)
print("mean pnl (%):",df['pnl'].mean(),"std pnl (%):",df['pnl'].std(),)
savedir = Path(in_args.savedir)
savedir.mkdir(parents=True, exist_ok=True)
file_name = savedir / (datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ ".csv")

df.to_csv(str(file_name))

if is_parallel:
    p.close()


