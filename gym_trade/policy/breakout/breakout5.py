from gym_trade.policy.base import BasePolicy
import numpy as np

class Policy(BasePolicy):
    def init_policy(self, **kwargs):
        Prv_High, Prv_Volme_High, Close, = kwargs['Prv_High'], kwargs['Prv_Volme_High'], kwargs['Close']
        self.calibrate_high_price = Prv_High * self.env.unwrapped.df['close'].iloc[-1] / Close
        self.calibrate_volume_high = Prv_Volme_High / (self.env.unwrapped.df['close'].iloc[-1] / Close)
        self.agg_volume = 0
        self.bk_or_prv = False
        self.high_close_price_after_buy = None
        self.close_price_buy = None
        self.break_high_prv = False
        self.pnl_prv = 0
        self.pnl_highest = 0
        self.buycnt = 0
        self.buysignal_cnt = 0
        self.buysignal_skipnum = 3 
        self.max_buycnt = 10
        self.max_loss = 0.02
        # self.max_timestep_pnl_drop = 1
        self.trade_curb_exist = False
        self.max_hold_timestep = 100
        self.hold_timestep_cnt = 0
        self.position_ratio_prv = 0
        self.price_limit_high = 100
        self.price_limit_low = 8
        self.start_open = None
        self.low_liquid = None
        self.start_trade_timestep = 1
        self.highest_prv = 0
        if self.gui:
            self.env.gui_init()
            self.env.gui_horizon_line(self.calibrate_high_price, text="previous high price")
            self.env.gui_textbox("symbol", self.env.file)

    def __call__(self, obs, **kwargs):
        if self.start_open is None:
            self.start_open = obs['open']

        if self.start_open is not None:
            if self.start_open < self.price_limit_low or self.start_open > self.price_limit_high: # price limit
                action = None
                return action
        
        if self.low_liquid is None:
            if obs['high'] == obs['low']:
                self.low_liquid = True
        if self.low_liquid is not None: 
            if self.low_liquid: # low liquit limit
                if obs['position_ratio']>0.1:
                    action = 1
                else:
                    action = None
                return action
            
        if self.position_ratio_prv<0.1 and obs['position_ratio']>0.1:
            self.close_price_buy = self.env.unwrapped.df['close'].iloc[self.env.timestep-1]
            self.high_close_price_after_buy = self.env.unwrapped.df['close'].iloc[self.env.timestep-1]
            self.buycnt+=1
        if self.position_ratio_prv>0.1 and obs['position_ratio']<0.1:
            self.high_close_price_after_buy = None
            self.close_price_buy = None
            self.hold_timestep_cnt = 0

        if obs['trade_curb']:
            self.trade_curb_exist = True

        self.agg_volume+=obs['volume']
        if obs["position_ratio"]>0.1:
            self.high_close_price_after_buy = np.maximum(obs['close'], self.high_close_price_after_buy)
            h2c = (obs['close'] - self.close_price_buy) / (self.high_close_price_after_buy - self.close_price_buy)
            pnl = (obs['close'] - self.close_price_buy) / self.close_price_buy
            self.pnl_highest = np.maximum(pnl, self.pnl_highest)
        else:
            h2c = 1
            pnl = 0
            self.pnl_highest = 0
        
        break_high = obs['close'] > self.highest_prv


       
        if  (break_high and not self.break_high_prv) \
            and obs["position_ratio"]<=0.1 \
            and self.env.timestep > 0 \
            and obs['close'] > self.calibrate_high_price\
            and self.agg_volume > self.calibrate_volume_high\
            and self.buycnt <self.max_buycnt\
            and  self.env.timestep >= self.start_trade_timestep  \
            and not self.trade_curb_exist\
                :
            self.buysignal_cnt+=1
            if self.buysignal_cnt >= self.buysignal_skipnum:
                action = 0 
            else:
                action = 2


        elif ((obs['timestep']>=388) 
            #   or ((pnl-self.pnl_prv) < (-self.max_loss * self.max_timestep_pnl_drop))
              or ( (self.pnl_highest-pnl) // self.max_loss -  (self.pnl_highest-self.pnl_prv) // self.max_loss) >= 1
            #   or ((pnl > 0.1) and (h2c<0.5))
            #   or (self.hold_timestep_cnt >= self.max_hold_timestep)
                # or not obs["break_high_or10"] 
                or (self.trade_curb_exist)
              ) \
             and obs["position_ratio"]>0.1:
            action =1
        else:
            action = 2
        
        if action == 1 and obs["position_ratio"]>0.1:
            self.hold_timestep_cnt +=1

        self.bk_or_prv = obs['break_high_or']
        self.break_high_prv = break_high
        self.position_ratio_prv = obs["position_ratio"]
        self.pnl_prv = pnl
        self.highest_prv = np.maximum(obs['high'],self.highest_prv)
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