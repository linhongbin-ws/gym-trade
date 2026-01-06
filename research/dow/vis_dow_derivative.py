import yfinance as yf
import os
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from lightweight_charts import Chart
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from gym_trade.tool import ta as ta_tool

def merge_ta_groups(cfg, prefix="ta_", out_key="ta", strict=True):
    merged = OmegaConf.create({})
    matched = []

    for k in cfg.keys():
        if str(k).startswith(prefix):
            v = cfg[k]
            matched.append((k, type(v).__name__))

            # 只允许 dict 类型
            if isinstance(v, DictConfig) or isinstance(v, dict):
                merged = OmegaConf.merge(merged, v)
            else:
                msg = f"[merge_ta_groups] Skip {k}: expected DictConfig, got {type(v).__name__}"
                if strict:
                    raise TypeError(msg)
                else:
                    print(msg)

    cfg[out_key] = merged
    print("[merge_ta_groups] matched keys:", matched)
    return cfg


@hydra.main(config_path="../../config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    cfg = merge_ta_groups(cfg, prefix="ta_", out_key="ta", strict=False)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # set proxy
    proxy = cfg.network.proxy
    os.environ['HTTP_PROXY'] = proxy 
    os.environ['HTTPS_PROXY'] = proxy 

    # download data
    if cfg.network.use_cache:
        df = pd.read_csv(os.path.join(cfg.network.cache_dir, 'dow.csv'))
    else:
        df = yf.download(['^DJI'], period='max',multi_level_index=False)
        if not os.path.exists(cfg.network.cache_dir):
            os.makedirs(cfg.network.cache_dir)
        df.to_csv(os.path.join(cfg.network.cache_dir, 'dow.csv'))

    df = standardlize_df(df)
    chart = Chart(toolbox=True,inner_width=1)
    chart.candle_style(down_color='#00ff55', up_color='#ed4807')
    chart.set(df)
    random_color = lambda : f'rgba({random.randint(100, 255)}, {random.randint(100, 255)}, {random.randint(100, 255)}, 0.9)'

    ta_dict = OmegaConf.to_container(
            cfg.ta,
            resolve=True,
            throw_on_missing=True,  # 有未填充的 ??? 就报错
            enum_to_str=True        # enum 转 string（结构化 config 时很有用）
        )
    for k, v in ta_dict.items():
        if k not in cfg.ta_select:
            continue
        _cfgs = v.copy()
        func = _cfgs['func']
        _cfgs.pop('func')
        call = getattr(ta_tool, func)
        results = call(df,**_cfgs)
        if isinstance(results, pd.DataFrame):
            for col in results.columns:
                df[k+'@'+col] = results[col]
            assert False, df.columns
        elif isinstance(results, pd.Series):
            df[k] = results
        else:
            raise NotImplementedError(f"Unsupported return type: {type(results)}")
        line_df = pd.DataFrame({
            'time': df.index,
            k: df[k]
        })
        line_df = line_df.dropna()
        line = chart.create_line(k, color = random_color(),)
        chart.legend(True)
        line.set(line_df)
        


    # chart.show(block=True)
    return None

if __name__ ==  '__main__':
    main()

# # set proxy
# proxy = 'http://127.0.0.1:7897'
# os.environ['HTTP_PROXY'] = proxy 
# os.environ['HTTPS_PROXY'] = proxy 


# if __name__ ==  '__main__':

#     df = yf.download(['^DJI'], period='max',multi_level_index=False)
#     print(df)
#     upsign = df['close'] > df['close'].shift(1)
#     change_sign = upsign ^ upsign.shift(1)

#     chart = Chart(toolbox=True,inner_width=1)
#     chart.candle_style(down_color='#00ff55', up_color='#ed4807')
#     chart.set(df)

    
#     df['chg_sign_close'] = df['close'].copy()
#     change_sign_shift = change_sign.shift(-1)
#     change_sign_shift.fillna(False)
#     change_sign_shift.iloc[-1] = True
#     change_sign_shift = change_sign_shift == True
#     print(change_sign_shift)
#     df['chg_sign_close'][~change_sign_shift] = np.nan
#     print(df['chg_sign_close'])
    
    
#     line = chart.create_line('chg_sign_close',color = 'rgba(100, 100, 255, 0.9)',)
#     chart.legend(True)

#     line_df = pd.DataFrame({
#         'time': df.index,
#         'chg_sign_close': df['chg_sign_close']
#     }).dropna()
#     line.set(line_df)
#     chart.show(block=True)

 