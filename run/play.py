from gym_trade.api import make_env, get_args
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm

args = get_args()
env, env_config = make_env(tags=args.env_tag, seed=args.seed)

if args.action == "oracle":
    env = ActionOracle(env, device=args.oracle_device)

for vis in args.vis:
    if vis == "chart":
        env =  LightChart_Visualizer(env, subchart_keys=args.lightchart_tag, keyboard=args.action != "oracle")
    elif vis == "image":
        env = CV_Visualizer(env, keyboard=False, vis_tag=args.vis_tag)

for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    print("reset timestep", env.timestep, )
    while not done:
        print("=============================")
        if any(i.isdigit() for i in args.action):
            action = int(args.action)
        elif args.action == "random":
            action = env.action_space.sample()
        elif args.action == "oracle":
            
            action = env.get_oracle_action()
        else:
            raise NotImplementedError
        obs, reward, done, info = env.step(action)
        print("Timestep:", env.timestep,)
        print_obs = obs.copy()
        print_obs = {k: v.shape if k in ["image","rgb","depth"] else v for k,v in print_obs.items()}
        print_obs = [str(k)+ ":" +str(v) for k,v in print_obs.items()]
        print("OBS:", " | ".join(print_obs))
    
        print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(),)
        imgs = env.render()
        imgs.update(obs)
        if "image" in args.vis:
            env.cv_show(imgs)
        if "chart" in args.vis:
            env.gui_show()