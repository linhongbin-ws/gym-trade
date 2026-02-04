import yfinance as yf
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from gym_trade.env.embodied import PaperTrade
from gym_trade.tool.get_data import load_data as load_data_func

def _load_test_daily_data():
    symbols = ["TSLA"]
    dfs  = load_data_func(symbols=symbols, interval="1d", start="2021-01-02", end='2024-01-05')
    return dfs

def test_env():
    dfs = _load_test_daily_data()
    env = PaperTrade(dfs, interval="1d")
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if action[0] <= 0:
            assert env.df.iloc[env.t]['dash@pos'] == env.df.iloc[env.t-1]['dash@pos']
            assert env.df.iloc[env.t]['dash@cash'] == env.df.iloc[env.t-1]['dash@cash']
    return env.pnl

if __name__ == "__main__":
    test_env()