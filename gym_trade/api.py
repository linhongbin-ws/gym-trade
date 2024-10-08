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


def screen_daily(daily_hdf, funcs, return_high=False):
    df_meta = pd.read_hdf(daily_hdf)
    symbols = df_meta.columns.levels[0]
    pbar = tqdm(symbols)
    # fil_func_strs = {'pre_gap': {'ratio_lower_bd':0.02}}
    results = {}
    for symbol in pbar:
        df = deepcopy(df_meta[symbol])
        df.dropna(inplace=True)
        if df.shape[0] <=1:
            continue
        for f in funcs:
            screen_func = getattr(screen, f[0])
            # print(f[1])
            screen_args = {k:v for k,v in f[1].items()}
            screen_args['df'] = df
            if f[0] == "new_high" and return_high:
                screen_args["return_high"] = True
                df, new_high = screen_func(**screen_args)
            else:
                df = screen_func(**screen_args)
                new_high = None
        if df is not None:
            results[symbol] = [pd.to_datetime(df.index), new_high]

    return results