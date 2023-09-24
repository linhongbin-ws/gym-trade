from gym_trade.env.wrapper.base import BaseWrapper  
from lightweight_charts import Chart

class LightChart_Visualizer(BaseWrapper):
    def __init__(self, 
                    env, 
                    **kwargs,
                     ):
        super().__init__(env)
        self._init_chart()

    def _init_chart(self):
        self._chart = Chart(toolbox=True)
        self._chart.legend(True)
        
    def render(self,):
        _df = self.unwrapped._df
        self._chart.set(_df.iloc[:self.unwrapped.timestep+1])
        self._chart.show(block=False)

    def close(self):
        self._chart.exit()