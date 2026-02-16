POLICY_REGISTRY = {}
FUNCTION_REGISTRY = {}

def register_policy(cls):
    # cls.__module__ → "gym_trade.policy.ma.simple_ma"
    module_path = cls.__module__.split(".")
    filename = module_path[-1]   # "simple_ma"

    if filename in POLICY_REGISTRY:
        raise KeyError(f"Duplicate policy name: {filename}")

    POLICY_REGISTRY[filename] = cls
    return cls


def register_function(fn):
    module_path = fn.__module__.split(".")
    filename = module_path[-1]   # "simple_ma"
    func_name = fn.__name__
    register_name = filename + "@" + func_name

    if register_name in FUNCTION_REGISTRY:
        raise KeyError(f"Duplicate function name: {register_name}")

    FUNCTION_REGISTRY[register_name] = fn
    return fn


