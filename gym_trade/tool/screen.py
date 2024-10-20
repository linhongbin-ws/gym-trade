


def price_limit(df, upper_limit, lower_limit):
    return df, (df['Close'] >= lower_limit)& (df['Close']<= upper_limit)



def new_high(df, period):
    """ filter pre-gap ratio with bound"""
    if df.shape[0] <=1: raise Exception("dataframe should have 2 or more rows")
    df['Prv_High'] = df.shift(1).rolling(period)['High'].max()
    # new_df = df[(df['High'] - _df_max)>0].copy()
    # new_max = _df_max[(df['High'] - _df_max)>0].copy()
    return df, (df['High'] -  df['Prv_High'])>0
    # if return_high:
    #     return new_df, new_max.values.tolist()
    # else:
    #     return new_df