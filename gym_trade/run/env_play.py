from gym_trade.api import make_env
from gym_trade.env.wrapper import LightChart_Visualizer
import argparse
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
parser.add_argument('--repeat',type=int, default=1)
parser.add_argument('--action',type=str, default="random")
parser.add_argument('--yaml-dir', type=str, default="./gym_trade/config/gym_trade.yaml")
parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
# if env_config.embodied_name == "dVRKEnv":
#     env =  Visualizer(env, update_hz=100)
# else:
env =  LightChart_Visualizer(env)
for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    # print("obs:", obs)
    while not done:
        # action = env.action_space.sample()
        # print(action)
        print("==========step", env.timestep, "===================")
        if any(i.isdigit() for i in args.action):
            action = int(args.action)
        elif args.action == "random":
            action = env.action_space.sample()
        elif args.action == "oracle":
            action = env.get_oracle_action()
        else:
            raise NotImplementedError
        print("step....")
        obs, reward, done, info = env.step(action)
        print_obs = obs.copy()
        print_obs = {k: v.shape if k in ["image","rgb","depth"] else v for k,v in print_obs.items()}
        print_obs = [str(k)+ ":" +str(v) for k,v in print_obs.items()]
        print(" | ".join(print_obs))
        print("reward:", reward, "done:", done,)
    
        # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
        # print("observation space: ", env.observation_space)
        char = env.gui_show()
        
        if char == 'q':
            break
