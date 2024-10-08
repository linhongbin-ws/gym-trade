





def new_high(df, period, return_high=False):
    """ filter pre-gap ratio with bound"""
    if df.shape[0] <=1: raise Exception("dataframe should have 2 or more rows")

    _df_max = df.shift(1).rolling(period)['High'].max()
    new_df = df[(df['High'] - _df_max)>0].copy()
    new_max = _df_max[(df['High'] - _df_max)>0].copy()
    if return_high:
        return new_df, new_max.values.tolist()
    else:
        return new_df