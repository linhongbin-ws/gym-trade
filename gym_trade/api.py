from gym_trade.tool.config import Config, load_yaml
from pathlib import Path
from gym_trade.env import wrapper as wp
import argparse
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from gym_trade.tool import screen


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-tag', type=str, nargs='+', default=[])
    parser.add_argument('--seed', type=int, default=0)   


    parser.add_argument('--vis', type=str, nargs='+', default=[])
    parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
    parser.add_argument('--lightchart-tag', type=str, nargs='+', default=[])

    parser.add_argument('--repeat',type=int, default=1)
    parser.add_argument('--action',type=str, default="oracle")
    parser.add_argument('--oracle-device', type=str, default='keyboard')
    args = parser.parse_args()
    return args


def make_env(env_config=None, tags=[], seed=0):
    assert isinstance(tags, list)
    if env_config is None:
        yaml_dir = Path( __file__ ).absolute().parent / "config" / "gym_trade.yaml"
        yaml_dict = load_yaml(yaml_dir)
        yaml_config = yaml_dict["default"].copy()
        config = Config(yaml_config)
    else:
        config = env_config
    # print(train_config)
    for tag in tags:
        config = config.update(yaml_dict[tag])
    if config.embodied_name == "GymTradeEnv":
        from gym_trade.env.embodied.gym_trade import GymTradeEnv
        _call = GymTradeEnv
    else:
        raise NotImplementedError
    # print(config)
    embodied_args = getattr(config.embodied, config.embodied_name)
    _kwargs ={}
    _kwargs["task"] = config.task_name
    task_name = config.task_name
    _kwargs.update(getattr(embodied_args, task_name).flat)
    for k,v in embodied_args.flat.items():
        if k.find(task_name)<0:
            _kwargs.update({k: v})
    env = _call(**_kwargs)

    for wrapper in config.wrapper.pipeline:
        if hasattr(wp, wrapper):
            _call = getattr(wp, wrapper)
            _kwargs =getattr(config.wrapper, wrapper).flat
            env = _call(env, **_kwargs)
    env.seed = seed
    config = config.update({"seed": seed})
    return env, config


def screen_daily(daily_hdf, config_screen, return_high=False, symbol_list=[]):
    df_meta = pd.read_hdf(daily_hdf)
    symbols = df_meta.columns.levels[0]
    pbar = tqdm(symbols)
    results = {}
    for symbol in pbar:
        if (not symbol in symbol_list) and (len(symbol_list)!=0):
            continue
        df = deepcopy(df_meta[symbol])
        df.dropna(inplace=True)
        if df.shape[0] <=1:
            continue
        df_bools = None
        new_column_names = ['Close']
        for f in config_screen.pipeline:
            screen_func = getattr(screen, f)
            screen_args = getattr(config_screen, f).flat
            # print(screen_args)
            screen_args['df'] = df
            df, df_bool, new_column_name = screen_func(**screen_args)
            df_bools = df_bool if df_bools is None else df_bools & df_bool
            if new_column_name is not None:
                new_column_names.append(new_column_name)
        
        df = df[df_bools]

        if len(df.index) == 0:
            continue

        df_results = {}
        df_results['dates'] = pd.to_datetime(df.index)
        for v in new_column_names:
            # print(v)
            df_results[v] = df[v].to_list()
        if df is not None:
            results[symbol] = df_results

    return results

def make_policy(policy_name, env):
    import importlib
    module_dir = 'gym_trade.policy.'+ policy_name
    module = importlib.import_module(module_dir)
    cls = getattr(module, 'Policy')
    return cls(env)


def backtest(env_tag, policy, minute_path, **kwargs):
    env, env_config = make_env(tags=env_tag, seed=0)
    env.load_stock_list([minute_path])
    obs = env.reset()

    p = make_policy(policy, env)
    p.init_policy(**kwargs)
    done = False
    while not done:
        action = p(obs)
        obs, reward, done, info = env.step(action)
    return env.pnl