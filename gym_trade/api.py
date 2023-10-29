from gym_trade.tool.config import Config, load_yaml
from pathlib import Path

def make_env(env_config=None, tags=[], seed=0):
    assert isinstance(tags, list)
    if env_config is None:
        yaml_dir = Path( __file__ ).absolute().parent / "config" / "gym_trade.yaml"
        yaml_dict = load_yaml(yaml_dir)
        yaml_config = yaml_dict["default"].copy()
        config = Config(yaml_config)
    else:
        config = env_config
    # print(train_config)
    for tag in tags:
        config = config.update(yaml_dict[tag])
    if config.client.name == "SurrolEnv":
        from gym_ras.env.embodied.surrol import SurrolEnv
        _call = SurrolEnv
    elif config.client.name == "dVRKEnv":
        _call = getattr(gym_ras.env.embodied.dvrk, config.client.name)
    else:
        raise NotImplementedError
    # print(config)
    _kwargs =getattr(config.client, config.client.name).flat
    task_config = getattr(config.client, config.client.task)
    _kwargs.update(task_config.flat)
    _kwargs.update({"task": config.client.task})
    env = _call(**_kwargs)

    for wrapper in config.wrapper.pipeline:
        if hasattr(gym_ras.env.wrapper, wrapper):
            _call = getattr(gym_ras.env.wrapper, wrapper)
            _kwargs =getattr(config.wrapper, wrapper).flat
            env = _call(env, **_kwargs)
    env.seed = seed
    config = config.update({"seed": seed})
    return env, config