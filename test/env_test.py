import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
args = parser.parse_args()

if args.p ==1:
    from gym_trade.env import US_Stock_Env
    env = US_Stock_Env(data_dir=str(Path(".") / "gym_trade" / "data" / "example"))
    env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.render()

