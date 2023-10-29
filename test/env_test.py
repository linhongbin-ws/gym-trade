import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
args = parser.parse_args()

if args.p ==1:
    from gym_trade.env import *
    env = US_Stock_Env(csv_root_dir=str(Path(".") / "gym_trade" / "data" / "example"))
    env = LightChart_Visualizer(env)
    env = Keyboard(env)
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("=====timestep: {}====".format(env.timestep))
        print_obs = [k+":"+ str(v) for k,v in obs.items()]
        print("OBS: ", "| ".join(print_obs), "ACTION:", action, "Reward:",reward)
        env.render()
        if info["keyboard"] == 'q':
            break

elif args.p ==2:
    from gym_trade.env import *
    env = US_Stock_Env(csv_root_dir=str(Path(".") / "gym_trade" / "data" / "example"))
    env = LightChart_Visualizer(env)
    env = Keyboard(env)
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("=====timestep: {}====".format(env.timestep))
        print_obs = [k+":"+ str(v) for k,v in obs.items()]
        print("OBS: ", "| ".join(print_obs), "ACTION:", action, "Reward:",reward)
        env.render()
        if info["keyboard"] == 'q':
            break

