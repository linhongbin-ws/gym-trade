from gym_trade.api import make_env, get_args
from gym_trade.env.wrapper import LightChart_Visualizer, ActionOracle,CV_Visualizer
from tqdm import tqdm

args = get_args()
env, env_config = make_env(tags=args.env_tag, seed=args.seed)

