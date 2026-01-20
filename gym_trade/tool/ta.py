import pandas as pd
from os.path import isfile
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
import numpy as np
from typing import List 
from gym_trade.tool import ta
from copy import deepcopy
import traceback
from typing import Tuple, Dict

def make_ta(df: pd.DataFrame, ta_dict_dict: dict[str, dict], col_range_key: dict| None = None,
         debug: bool= True) -> pd.DataFrame: 

    # input df will have columns of ['date', 'close', 'high', 'low', 'open', 'volume']
    if col_range_key is None:
        col_range_key = {'close': [0, np.inf], 
                        'high': [0, np.inf],  
                        'low': [0, np.inf],  
                        'open': [0, np.inf],  
                        'volume': [0, np.inf], }
        

    unfinish_dict = {}
    for ta_name, ta_arg_dict in ta_dict_dict.items():
        func = ta_arg_dict['func']
        call = globals()[func]
        args = {k: v for k, v in ta_arg_dict.items() if k != 'func' }
        try:
            args['key_range'] = col_range_key[args['key']]
            ta_results, ta_ranges = call(df, **args)
        except Exception as e:
            if debug:
                traceback.print_exc()
            unfinish_dict[ta_name] = ta_arg_dict
            continue
        if isinstance(ta_results, pd.DataFrame):
            for k in ta_results.columns:
                name = ta_name + '@' + k
                df[name] = ta_results[k]
                col_range_key[name] = ta_ranges[k]
        else:
            raise NotImplementedError(f"Unsupported return type: {type(ta_results)}") 
    return df, col_range_key, unfinish_dict, 


def _rolling_conv(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # convolution aligned to "window ends": result length = len(y) - len(w) + 1
    return np.convolve(y, w[::-1], mode="valid")

def range_mapping(key_range: List[float], map_type: str = "identity"):
    assert len(key_range) == 2 and key_range[0] < key_range[1], f"{key_range} not eligible range"

    if map_type == "identity":
        out_range = deepcopy(key_range)
    else:
        raise NotImplementedError
    return out_range


##====================TA s============================

def ma(df: pd.DataFrame, key:str, key_range: List[float],  window:int) -> pd.Series:
    series = df[key].copy()
    out = {"value": series.rolling(window, min_periods=window).mean()}
    out_range = {"value": range_mapping(key_range)}
    return pd.DataFrame(out, index = series.index), out_range


def rolling_linreg_features(df: pd.DataFrame, key:str, key_range: List[float], window: int) -> pd.DataFrame:
    """
    Rolling OLS on fixed x = 0..window-1.
    Returns slope, intercept, r2, tstat, slope_pct_annual (annualized slope / level).
    """
    series = df[key].copy()
    y = series.to_numpy(dtype=float)
    n = len(y)
    w = int(window)
    out = {
        "slope": np.full(n, np.nan),
        "intercept": np.full(n, np.nan),
        "r2": np.full(n, np.nan),
        "t": np.full(n, np.nan),
        "slope_pct_annual": np.full(n, np.nan),
    }
    if w < 3 or n < w:
        return pd.DataFrame(out, index=series.index)

    x = np.arange(w, dtype=float)
    sumx = x.sum()
    sumx2 = (x * x).sum()
    # centered variance term
    Sxx = sumx2 - (sumx * sumx) / w
    denom = w * sumx2 - sumx * sumx  # = w*Sxx

    # rolling sums via convolution
    ones = np.ones(w, dtype=float)
    sumy = _rolling_conv(y, ones)
    sumy2 = _rolling_conv(y * y, ones)
    sumxy = _rolling_conv(y, x)

    # slope & intercept
    slope = (w * sumxy - sumx * sumy) / denom
    intercept = (sumy - slope * sumx) / w

    # SSE via: SSE = y'y - 2 beta'X'y + beta'(X'X)beta
    # X'X constants
    XTX_00 = w
    XTX_01 = sumx
    XTX_11 = sumx2
    # X'y rolling
    XTy_0 = sumy
    XTy_1 = sumxy
    # beta terms
    b0 = intercept
    b1 = slope

    beta_XTy = b0 * XTy_0 + b1 * XTy_1
    beta_XTX_beta = (b0 * b0) * XTX_00 + 2 * (b0 * b1) * XTX_01 + (b1 * b1) * XTX_11
    SSE = sumy2 - 2.0 * beta_XTy + beta_XTX_beta

    # SST = sum((y - ybar)^2) = y'y - (sumy^2)/w
    SST = sumy2 - (sumy * sumy) / w
    # avoid divide-by-zero
    r2 = np.where(SST > 0, 1.0 - (SSE / SST), np.nan)

    # t-stat for slope: se(b1) = sqrt( (SSE/(w-2))/Sxx )
    mse = SSE / (w - 2)
    se_b1 = np.sqrt(np.where((mse > 0) & (Sxx > 0), mse / Sxx, np.nan))
    tstat = slope / se_b1

    # annualized slope normalized by level (end value of window)
    # end level for each window is series aligned to window end
    level_end = series.to_numpy(dtype=float)[w-1:]
    slope_pct_annual = (slope / level_end) * 252.0  # per-year

    # place into output aligned to window end
    idx = np.arange(w - 1, n)
    out["slope"][idx] = slope
    out["intercept"][idx] = intercept
    out["r2"][idx] = r2
    out["t"][idx] = tstat
    out["slope_pct_annual"][idx] = slope_pct_annual

    out_range = {"slope": [-np.inf,np.inf ],
                "intercept": [-np.inf,np.inf ],
                "r2": [0,1],
                "t":  [-np.inf,np.inf ],
                "slope_pct_annual": [-np.inf,np.inf ],
                }

    return pd.DataFrame(out, index=series.index), out_range




def ema(df: pd.DataFrame, key:str, key_range: List[float], window: int) -> pd.Series:
    series = df[key].copy()
    out = {"value": series.ewm(span=window, adjust=False, min_periods=window).mean()}
    out_range = {"value": range_mapping(key_range)}
    return pd.DataFrame(out, index = series.index), out_range



def rsi_momentum_features(
    df: pd.DataFrame,
    key: str,
    key_range: List[float],
    rsi_window: int = 14,
    mom_periods: List[int] = [1, 5, 15],
    ema_fast: int = 12,
    ema_slow: int = 48,
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    series = df[key].astype(float).copy()

    # RSI (Wilder-style via EMA on gains/losses)
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/rsi_window, adjust=False, min_periods=rsi_window).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/rsi_window, adjust=False, min_periods=rsi_window).mean()

    rs = gain_ema / (loss_ema + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # momentum returns
    mom = {}
    for p in mom_periods:
        mom[f"mom_{p}"] = series.pct_change(p)

    # EMA trend
    ef = series.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    es = series.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    ema_diff_norm = (ef - es) / (series.abs() + eps)

    out = {
        "rsi": rsi,
        "ema_fast": ef,
        "ema_slow": es,
        "ema_diff_norm": ema_diff_norm,
        **mom,
    }

    out_range = {
        "rsi": [0.0, 100.0],
        "ema_fast": [-np.inf, np.inf],
        "ema_slow": [-np.inf, np.inf],
        "ema_diff_norm": [-np.inf, np.inf],
    }
    for p in mom_periods:
        out_range[f"mom_{p}"] = [-np.inf, np.inf]

    return pd.DataFrame(out, index=series.index), out_range



def bollinger_features(
    df: pd.DataFrame,
    key: str,
    key_range: List[float],
    window: int,
    n_std: float = 2.0,
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    series = df[key].astype(float).copy()

    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()

    upper = mid + n_std * std
    lower = mid - n_std * std

    bandwidth = (upper - lower) / (mid.abs() + eps)
    pct_b = (series - lower) / ((upper - lower) + eps)
    z = (series - mid) / (std + eps)

    out = {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
        "z": z,
    }

    out_range = {
        "mid": [-np.inf, np.inf],
        "upper": [-np.inf, np.inf],
        "lower": [-np.inf, np.inf],
        "bandwidth": [0.0, np.inf],
        "pct_b": [-np.inf, np.inf],   # often ~[0,1], but can exceed if price breaks bands
        "z": [-np.inf, np.inf],
    }

    return pd.DataFrame(out, index=series.index), out_range



def atr_features(
    df: pd.DataFrame,
    key_range: List[float],
    window: int,
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(window, min_periods=window).mean()

    atr_norm = atr / (close.abs() + eps)
    range_norm = (high - low) / (close.abs() + eps)
    gap_norm = (open_ - prev_close) / (prev_close.abs() + eps)

    out = {
        "tr": tr,
        "atr": atr,
        "atr_norm": atr_norm,
        "range_norm": range_norm,
        "gap_norm": gap_norm,
    }

    out_range = {
        "tr": [0.0, np.inf],
        "atr": [0.0, np.inf],
        "atr_norm": [0.0, np.inf],
        "range_norm": [0.0, np.inf],
        "gap_norm": [-np.inf, np.inf],
    }

    return pd.DataFrame(out, index=df.index), out_range


def candle_structure_features(
    df: pd.DataFrame,
    key_range: List[float],
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = c - o
    bar_range = (h - l).abs() + eps

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    body_norm = body / (c.abs() + eps)
    upper_wick_norm = upper_wick / bar_range
    lower_wick_norm = lower_wick / bar_range
    close_pos = (c - l) / bar_range
    is_bull = (c > o).astype(float)

    out = {
        "body": body,
        "body_norm": body_norm,
        "upper_wick_norm": upper_wick_norm,
        "lower_wick_norm": lower_wick_norm,
        "close_pos": close_pos,
        "is_bull": is_bull,
    }

    out_range = {
        "body": [-np.inf, np.inf],
        "body_norm": [-np.inf, np.inf],
        "upper_wick_norm": [0.0, 1.0],
        "lower_wick_norm": [0.0, 1.0],
        "close_pos": [0.0, 1.0],
        "is_bull": [0.0, 1.0],
    }

    return pd.DataFrame(out, index=df.index), out_range



def vwap_features(
    df: pd.DataFrame,
    key_range: List[float],
    price_key: str = "close",
    dev_z_window: int = 60,
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    price = df[price_key].astype(float)
    vol = df["volume"].astype(float)

    # group by date, reset each session
    date = df.index.date
    pv = price * vol
    cum_pv = pv.groupby(date).cumsum()
    cum_vol = vol.groupby(date).cumsum()
    vwap = cum_pv / cum_vol.replace(0.0, np.nan)

    dev = (price - vwap) / (vwap.abs() + eps)

    # rolling zscore of dev (for mean-reversion thresholds)
    dev_mean = dev.rolling(dev_z_window, min_periods=dev_z_window).mean()
    dev_std = dev.rolling(dev_z_window, min_periods=dev_z_window).std()
    dev_z = (dev - dev_mean) / (dev_std + eps)

    out = {
        "vwap": vwap,
        "dev": dev,
        "dev_z": dev_z,
    }
    out_range = {
        "vwap": [-np.inf, np.inf],
        "dev": [-np.inf, np.inf],
        "dev_z": [-np.inf, np.inf],
    }

    return pd.DataFrame(out, index=df.index), out_range


def volume_features(
    df: pd.DataFrame,
    key_range: List[float],
    window: int,
    eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    vol = df["volume"].astype(float)

    vol_mean = vol.rolling(window, min_periods=window).mean()
    vol_std = vol.rolling(window, min_periods=window).std()

    vol_ratio = vol / (vol_mean + eps)
    vol_z = (vol - vol_mean) / (vol_std + eps)

    # reuse your linreg helper for a "volume trend" feature
    lin_df, _ = rolling_linreg_features(df.assign(_vol=vol), key="_vol", key_range=[0, 1], window=window)
    vol_slope = lin_df["slope"]

    out = {
        "vol_ratio": vol_ratio,
        "vol_z": vol_z,
        "vol_slope": vol_slope,
    }
    out_range = {
        "vol_ratio": [0.0, np.inf],
        "vol_z": [-np.inf, np.inf],
        "vol_slope": [-np.inf, np.inf],
    }

    return pd.DataFrame(out, index=df.index), out_range





#============= draft ===========================

def break_high(df: pd.DataFrame, key, ):
    _df = df.copy()
    _df['new_high'] = np.nan
    _new_high = None
    for ind in _df.index:
        if _new_high is None:
            _new_high = _df[key][ind]
            value = 1
        elif _new_high <  _df[key][ind]:
            _new_high = _df[key][ind]
            value = 1
        else:
            value = 0
            
        _df['new_high'][ind] = value
    return _df['new_high']

def rolling_op(df, key, operation, window):
    _df = df.copy()
    if operation == "or":
        seri = _df[key].rolling(window=window).max().astype(bool) # equivalent to or operation
    else:
        raise NotImplementedError
    return seri

def direction(df, key):
    _df = df.copy()
    seri = (_df[key] - _df[key].shift(1))>0
    return seri

def trade_curb(df):
    _df = df.copy()
    seri1 = _df['high'] == _df['low']
    seri2 = _df['volume'] <=0
    seri = seri1 & seri2
    return seri



def direction_toggle(df, key):
    # direction of a key is change and it will give True, otherwise False
    _df = df.copy()
    upsign = _df[key] > _df[key].shift(1)
    change_sign = upsign ^ upsign.shift(1)
    change_sign_shift = change_sign.shift(-1)
    change_sign_shift.fillna(False)
    change_sign_shift.iloc[-1] = True

    return_list = {}
    return_list['direction_toggle_bool@'+key] =  np.int32(change_sign_shift) # 1 or 0
    return_list['direction_toggle_value@'+key] =  _df[key].copy()
    return_list['direction_toggle_value@'+key][np.logical_not(change_sign_shift)] = np.nan

    # pattern id
    # 0 : index <3

    # pattern 1
    #       ^
    #   ^  
    #     v
    # v 

    # pattern 2
    #       ^
    #   ^  
    # v    
    #     v

    # pattern 3
    #   ^    
    #       ^
    #     v 
    # v    

    # pattern 4
    #   ^    
    #       ^
    # v     
    #     v

    # pattern 5
    #   ^    
    # v      
    #       ^
    #     v      

    # pattern 6
    # ^      
    #     ^  
    #   v     
    #       v 

    # pattern 7
    #     ^   
    # ^      
    #   v     
    #       v
 

    # pattern 8
    # ^      
    #     ^   
    #       v  
    #   v    

    # pattern 9
    #     ^   
    # ^      
    #       v  
    #   v   

    # pattern 10
    #     ^   
    #       v 
    # ^        
    #   v    
    pattern_sr = return_list['direction_toggle_value@'+key].copy()
    pattern_sr[:] = np.nan
    for pt_idx, _ in enumerate(pattern_sr):
        new_seri = return_list['direction_toggle_value@'+key][:pt_idx+1].dropna().copy()
        if len(new_seri.index) < 4:
            pt_value = 0
        else:
            if np.isnan(return_list['direction_toggle_value@'+key][pt_idx]):
                pivot = new_seri.iloc[-3:].tolist()
            else:
                pivot = new_seri.iloc[-4:-1].tolist()
            pivot.append(_df[key][pt_idx])
            if pivot[3]>pivot[2]:
                if pivot[3]>pivot[1] and pivot[2]>pivot[0]:
                    # date = pattern_sr.index[pt_idx]
                    pt_value = 1
                elif pivot[3]>pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 2
                elif pivot[3]<pivot[1] and pivot[2]>pivot[0]:
                    pt_value = 3
                elif pivot[3]<pivot[1] and pivot[2]<pivot[0] and pivot[3]>pivot[0]:
                    pt_value = 4
                else:
                    pt_value = 5
            else:
                if pivot[3]<pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 6
                elif pivot[3]<pivot[1] and pivot[2]>pivot[0]:
                    pt_value = 7
                elif pivot[3]>pivot[1] and pivot[2]<pivot[0]:
                    pt_value = 8
                elif pivot[3]>pivot[1] and pivot[2]>pivot[0] and pivot[3]<pivot[0]:
                    pt_value = 9
                else:
                    pt_value = 10
        pattern_sr.iloc[pt_idx] = pt_value
    
    return_list['direction_toggle_pattern_id@'+key] = pattern_sr
    return_list['direction_toggle_pattern_strongup@'+key] = np.int32(pattern_sr ==1)

    strong_up_acc = return_list['direction_toggle_value@'+key].copy()
    strong_up_acc[:] = np.nan
    cnt = 0
    for idx, _ in enumerate(strong_up_acc):
        if idx == 0:
            strong_up_acc[idx] = 0
        else:
            if pattern_sr[idx] in [1] and  pattern_sr[idx] !=  pattern_sr[idx-1]:
                cnt +=1
            elif pattern_sr[idx] in [6,7,8]:
                cnt=0
            strong_up_acc[idx] = cnt
    return_list['direction_toggle_pattern_strongup_acc@'+key] = strong_up_acc

    return return_list