# policy/registry.py
POLICY_REGISTRY = {}

def register_policy(cls):
    # cls.__module__ → "gym_trade.policy.ma.simple_ma"
    module_path = cls.__module__.split(".")
    filename = module_path[-1]   # "simple_ma"

    if filename in POLICY_REGISTRY:
        raise KeyError(f"Duplicate policy name: {filename}")

    POLICY_REGISTRY[filename] = cls
    return cls
