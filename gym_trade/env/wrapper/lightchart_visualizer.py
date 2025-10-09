from gym_trade.env.wrapper.base import BaseWrapper  
from lightweight_charts import Chart
import pandas as pd
from gym_trade.tool.keyboard import Keyboard
import sys
class LightChart_Visualizer(BaseWrapper):
    def __init__(self, 
                    env, 
                    subchart_keys=[],
                    keyboard = True,
                    plot_toggle_direction=True,
                    **kwargs,
                     ):
        super().__init__(env)
        self._subchart_keys = subchart_keys
        self._keyboard = Keyboard() if keyboard else None
        self._plot_toggle_direction = plot_toggle_direction
        self._lines = {}
        self.gui_init()
    def gui_init(self):
        self._init_chart()
    def _init_chart(self):
        main_chart_width = 0.6
        self._chart = Chart(toolbox=True,inner_width=1, inner_height=main_chart_width)
        self._chart.candle_style(down_color='#00ff55', up_color='#ed4807')
        self._chart.legend(True)
        self._sub_charts = {}
        # if len(self._subchart_keys)==0:
        #      subchart_keys = list(self.unwrapped.df.columns.values)
        #      self._subchart_keys = [k for k in subchart_keys if k not in ['date', 'open', 'high', 'low', 'close', 'volume',]]
        for _key in self._subchart_keys:
            self._sub_charts[_key] = [None, None]
            self._sub_charts[_key][0] = self._chart.create_subchart(position='left', width=1, height=(1-main_chart_width)/len(self._subchart_keys),sync=True)
            self._sub_charts[_key][0].legend(True)
            self._sub_charts[_key][1] = self._sub_charts[_key][0].create_line(_key)

    def gui_horizon_line(self, value, text=''):
        self._chart.horizontal_line(value,text=text)
    def gui_marker(self, maker_name):
        _df = self.unwrapped.df
        self._chart.marker(text=maker_name, time=_df.index[self.unwrapped.timestep])
    def gui_textbox(self, text_type, text):
        self._chart.topbar.textbox(text_type, text)

    def gui_show(self,):
        _df = self.unwrapped.df.copy()
        
        self._chart.set(_df.iloc[:self.unwrapped.timestep+1])


        if self._plot_toggle_direction:
            for k in _df:
                if k.find('direction_toggle_value')!=-1:
                    if not k in self._lines:
                        self._lines[k] = self._chart.create_line(k,color = 'rgba(100, 100, 255, 0.9)',)
                    line_df = pd.DataFrame({
                        'time': _df.index,
                        k: _df[k]
                    }).dropna()
                    self._lines[k].set(line_df)


        for k,v in self._sub_charts.items():
            _data = _df[[k]].iloc[:self.unwrapped.timestep+1]
            # _data = pd.DataFrame({
            #         'time': _df.index[k].iloc[:self.unwrapped.timestep+1],
            #         _key: _df[k].iloc[:self.unwrapped.timestep+1]})
            v[1].set(_data)


                    

        

        if self._keyboard is not None:
            self._chart.show(block=False)
            char = self._keyboard.get_char()
            if char == 'q':
                sys.exit(0)
        else:
            self._chart.show(block=True)
            char = None
        return char

    def __del__(self):
        self._chart.exit()

