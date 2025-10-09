from gym_trade.env.embodied.gym_trade import GymTradeEnv
import yfinance as yf
import os
import datetime
from gym_trade.env.wrapper import LightChart_Visualizer, TA
if __name__ ==  '__main__':


    proxy = 'http://127.0.0.1:7897'
    os.environ['HTTP_PROXY'] = proxy 
    os.environ['HTTPS_PROXY'] = proxy 

    # df = yf.download(['BRK-B'], period='max',multi_level_index=False)
    # df = yf.download(['TSLA'], period='max',multi_level_index=False)
    # df = yf.download(['QQQ'], period='max',multi_level_index=False)
    df = yf.download(['EDU'], period='max',multi_level_index=False)

    df = df.truncate(before=datetime.datetime(2020, 1, 1),)
    df_list = [df]
    env = GymTradeEnv(task="us_stock",
                    df_list=df_list, 
                        init_balance=1000000,
                        commission_type = "free", 
                        reward_type='pnl_delta_sparse',
                        obs_keys=["position_ratio","open","close","high","low"],
                        stat_keys=['stat_pos', 'position_ratio', 'stat_pnl','stat_balance','stat_cash',],
                        action_min_thres=0.1,
                        fix_buy_position=True,
                        interval="day",)

    

    #ta
    ta_dict_list = []
    ta_dict_list.append({"func":"direction_toggle", "key":"close"})
    

    env = TA(env, ta_dict_list=ta_dict_list)


    
    obs = env.reset()
    env = LightChart_Visualizer(env,keyboard=True,subchart_keys=['direction_toggle_bool@close',
                                                                 'direction_toggle_pattern_id@close',
                                                                 'direction_toggle_pattern_strongup@close',
                                                                 'direction_toggle_pattern_strongup_acc@close',
                                                                 'stat_pnl'])
    done = False
    sig_cnt = 0
    sig_cnt_thres =1
    while not done:
        if obs['direction_toggle_pattern_strongup_acc@close'] >=1:
            sig_cnt+=1
        else:
            sig_cnt=0

        if obs['position_ratio'] <0.9 and sig_cnt>=sig_cnt_thres:
            action = 1
        elif obs['position_ratio'] >0.9 and sig_cnt<sig_cnt_thres:
            action = -1
        else:
            action = -0

        if action>=0.1:
            env.gui_marker("buy")
            # print(f"buy at timestep {env.timestep}")
        if action<=-0.1:
            env.gui_marker("sell")
            print(f"sell at timestep {env.timestep}")
        obs, reward, done, info = env.step(action)
        print(f"time: {env.timestep}/ {len(env.df.index)-1}. reward: {reward}, pnl: {env.pnl}")

    env.gui_show()
    # from lightweight_charts import Chart
    # chart = Chart(toolbox=True,inner_width=1)
    # chart.candle_style(down_color='#00ff55', up_color='#ed4807')
    # chart.set(env.unwrapped.df)
    # chart.show(block=True)

