from gym_trade.env.wrapper.base import BaseWrapper  

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
pd.plotting.register_matplotlib_converters() # fix bug in windows for pyplot
matplotlib.use('TKAgg')
import mplfinance as mpf
pd.options.mode.chained_assignment = None  # disable warning for pandas

class Mfinance_Visualizer(BaseWrapper):
    def __init__(self, 
                    env, 
                    **kwargs,
                     ):
        super().__init__(env)
         
    def render(self, 
                mode = 'human', 
                plot_features_dict={}):
        assert mode == 'human', f'{mode} is not support'

        _plot_features_dict = {feature_name:None for feature_name in self.feature_names}
        _plot_features_dict.update(plot_features_dict)
        style.use('ggplot')
        _df = self._df.iloc[:min(self._timestep+1, self._df.shape[0])].copy()
        df_mpf = _df[['open','high','low','close','volume']].copy()
        df_mpf = df_mpf.rename(columns={"open": "Open",
                                        "high": "High",
                                        "low": "Low",
                                        "close": "Close",
                                        "volume": "Volume"}) # names for mplfinance
        df_mpf.index.name = 'Date'
        panel_idx = 1
        ap =[]
        ts = min(self._timestep+1, self._df.shape[0])
        buy_actions = self._df['close'].iloc[:ts].copy() *0.98
        sell_actions = self._df['close'].iloc[:ts].copy() *1.02
        # print(self._df['is_hold'].iloc[:ts]==True)
        buy_actions[~((self._df['is_hold'].iloc[:ts]==True) & (self._df['is_hold'].shift(1).iloc[:ts]==False))] = np.nan
        sell_actions[~((self._df['is_hold'].iloc[:ts]==False) & (self._df['is_hold'].shift(1).iloc[:ts]==True))] = np.nan
        buy_actions = buy_actions.to_list()
        sell_actions = sell_actions.to_list()
        # print(np.sum(buy_actions))
        if not np.isnan(buy_actions).all():
            ap.append(mpf.make_addplot(buy_actions, type='scatter', panel=0,
                                color = 'red', markersize=50, marker='^'))
        if not np.isnan(sell_actions).all():
            ap.append(mpf.make_addplot(sell_actions, type='scatter', panel=0,
                                color = 'green', markersize=50, marker='v'))    
        for k, v in _plot_features_dict.items():
            if v is None:
                panel_idx +=1
                _panel_idx = panel_idx
            else:
                _panel_idx = v

            ap.append(mpf.make_addplot(self._df[k].iloc[:ts], panel=_panel_idx,
                                    type='line', ylabel=k))
        mpf.plot(df_mpf, addplot = ap, volume = True, type='candle',mav=(3), title=self._csv_list[self._csv_idx])
