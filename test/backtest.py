from gym_trade.api import make_env, get_args
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm

env, env_config = make_env(tags=[], seed=0)

file = "~/ssd/data/stock-data/us-minute/kminute-2024-10-02/2024-10-02-AAPL.csv"
env.load_stock_list([file])
obs = env.reset()
print(env.file)
done = False
while not done:
    if obs['timestep'] == 1:
        action = 0
    elif obs['timestep'] == 389:
        action = 1
    else:
        action = 2
    obs, reward, done, info = env.step(action)
    obs.pop('rgb', None)
    obs.pop('image', None)
    print("obs", obs)
print(env.pnl)
