import pandas as pd
from lightweight_charts import Chart
from gym_trade.tool.preprocess import fill_missing_frame, standardlize_df
import time

if __name__ == '__main__':

    _dir = './gym_trade/data/example/kminute-2021-01-05/2021-01-05-AA.csv'
    chart = Chart()
    
    # Columns: time | open | high | low | close | volume 
    df = pd.read_csv(_dir)
    print(df)

    df = standardlize_df(df)
    df = fill_missing_frame(df)# filling missing frame
    chart.set(df)
    
    chart.show()
    time.sleep(2)
    print("xxx")
    chart.set(df)
    
    chart.show()

    time.sleep(2)
