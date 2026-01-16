from omegaconf import DictConfig, OmegaConf


def merge_ta_groups(cfg, prefix="ta_", out_key="ta", strict=True):
    merged = OmegaConf.create({})
    matched = []

    for k in cfg.keys():
        if str(k).startswith(prefix):
            v = cfg[k]
            matched.append((k, type(v).__name__))

            # 只允许 dict 类型
            if isinstance(v, DictConfig) or isinstance(v, dict):
                merged = OmegaConf.merge(merged, v)
            else:
                msg = f"[merge_ta_groups] Skip {k}: expected DictConfig, got {type(v).__name__}"
                if strict:
                    raise TypeError(msg)
                else:
                    print(msg)

    cfg[out_key] = merged
    print("[merge_ta_groups] matched keys:", matched)
    return cfg
