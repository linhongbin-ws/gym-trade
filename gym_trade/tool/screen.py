


def price_limit(df, upper_limit, lower_limit):
    return df, (df['Close'] >= lower_limit)& (df['Close']<= upper_limit), None



def new_high(df, period):
    if df.shape[0] <=1: raise Exception("dataframe should have 2 or more rows")
    df['Prv_High'] = df.shift(1).rolling(period)['High'].max()
    return df, (df['High'] -  df['Prv_High'])>0, "Prv_High"



def new_volume_high(df, period):
    """ filter pre-gap ratio with bound"""
    if df.shape[0] <=1: raise Exception("dataframe should have 2 or more rows")
    df['Prv_Volme_High'] = df.shift(1).rolling(period)['Volume'].max()
    return df, (df['Volume'] -  df['Prv_Volme_High'])>0, "Prv_Volme_High"