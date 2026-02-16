from gym_trade.policy.base import BasePolicy
from gym_trade.policy.registry import register_policy

@register_policy
class Policy(BasePolicy):
    def init_policy(self, **kwargs):
        Prv_High, Prv_Volme_High, Close, = kwargs['Prv_High'], kwargs['Prv_Volme_High'], kwargs['Close']
        self.calibrate_high_price = Prv_High * self.env.unwrapped.df['close'].iloc[-1] / Close
        self.calibrate_volume_high = Prv_Volme_High / (self.env.unwrapped.df['close'].iloc[-1] / Close)
        self.agg_volume = 0
        self.bk_or_prv = False
        if self.gui:
            self.env.gui_init()
            self.env.gui_horizon_line(self.calibrate_high_price, text="previous high price")
            self.env.gui_textbox("symbol", self.env.file)

    def __call__(self, obs, **kwargs):
        self.agg_volume+=obs['volume']
        if obs['break_high'] \
            and obs["position_ratio"]<=0.1 \
            and self.env.timestep > 0 \
            and obs['high'] > self.calibrate_high_price\
            and self.agg_volume > self.calibrate_volume_high\
                :
            action = 0 
        elif obs['timestep']>=388 and obs["position_ratio"]>0.1:
            action =1
            # print("sell when market close")
        elif (self.bk_or_prv and (not obs['break_high_or'])) and obs["position_ratio"]>0.1:
            action =1
            # print("sell")
        else:
            action = 2


        self.bk_or_prv = obs['break_high_or']
        return action
        

# def backtest(minute_path, Prv_High, Prv_Volme_High, Close, gui=False,timeout=10, **kwargs):
#     if timeout>0:
#         start_time = time.time()
#     env, env_config = make_env(tags=[], seed=0)
#     if gui:
#         env = LightChart_Visualizer(env)
#     env.load_stock_list([minute_path])
#     file_name = Path(minute_path).stem
#     obs = env.reset()
#     calibrate_high_price = Prv_High * env.unwrapped.df['close'].iloc[-1] / Close
#     calibrate_volume_high = Prv_Volme_High / (env.unwrapped.df['close'].iloc[-1] / Close)
#     if gui:
#         env.gui_init()
#         env.gui_horizon_line(calibrate_high_price, text="previous high")
#         env.gui_textbox("symbol", file_name)
#     done = False
#     new_high = obs['high']
#     break_keep_count = 0
#     max_break_keep = 2
#     # print(obs.keys())
#     agg_volume = obs['volume']
#     while not done:
#         if timeout >0 and (time.time()- start_time > timeout):
#             print("timeout..")
#             return None

#         if obs['break_high'] \
#             and obs["position_ratio"]<=0.1 \
#             and env.timestep > 0 \
#             and obs['high'] > calibrate_high_price\
#             and agg_volume > calibrate_volume_high\
#                 :
#             action = 0 
#         elif obs['timestep']>=388 and obs["position_ratio"]>0.1:
#             action =1
#             # print("sell when market close")
#         elif (bk_or_prv and (not obs['break_high_or'])) and obs["position_ratio"]>0.1:
#             action =1
#             # print("sell")
#         else:
#             action = 2
        
#         bk_or_prv = obs['break_high_or']
#         obs, reward, done, info = env.step(action)
#         agg_volume+=obs['volume']
#         # print(env.unwrapped.timestep, action)
#         if gui:
#             if action==0:
#                 env.gui_marker("buy")
#                 print(f"buy at timestep {env.timestep}")
#             if action==1:
#                 env.gui_marker("sell")
#                 print(f"sell at timestep {env.timestep}")
#     if gui:
#         env.gui_textbox("pnl", "pnl: " + str(env.pnl))
#         env.gui_show()
#     return env.pnl