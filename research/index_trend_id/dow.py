import yfinance as yf
import os
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
from lightweight_charts import Chart
from gym_trade.tool.keyboard import Keyboard
import numpy as np
import pandas as pd
if __name__ ==  '__main__':
    proxy = 'http://127.0.0.1:7897'
    os.environ['HTTP_PROXY'] = proxy 
    os.environ['HTTPS_PROXY'] = proxy 
    # dat = yf.Ticker("MSFT")
    keyboard = Keyboard()
    # df = yf.download(['^DJI'], period='max',multi_level_index=False)
    df = yf.download(['BRK-B'], period='max',multi_level_index=False)
    print(df)
    df = standardlize_df(df) 
    print(df)
    upsign = df['close'] > df['close'].shift(1)
    change_sign = upsign ^ upsign.shift(1)

    chart = Chart(toolbox=True,inner_width=1)
    chart.candle_style(down_color='#00ff55', up_color='#ed4807')
    chart.set(df)

    
    df['chg_sign_close'] = df['close'].copy()
    change_sign_shift = change_sign.shift(-1)
    change_sign_shift.fillna(False)
    change_sign_shift.iloc[-1] = True
    change_sign_shift = change_sign_shift == True
    print(change_sign_shift)
    df['chg_sign_close'][~change_sign_shift] = np.nan
    print(df['chg_sign_close'])
    
    
    line = chart.create_line('chg_sign_close',color = 'rgba(100, 100, 255, 0.9)',)
    chart.legend(True)

    line_df = pd.DataFrame({
        'time': df.index,
        'chg_sign_close': df['chg_sign_close']
    }).dropna()
    line.set(line_df)
    chart.show(block=True)

 