"""
[Archived 2026-05-19]

The direction-toggle strong-up baseline that used to live here was rewritten as a
first-class policy. Use it through the standard main.py pipeline:

    .venv/bin/python main.py policy=direction_toggle data.symbol=[EDU] \\
        data.interval=1d data.start=2020-01-01 mode.name=bt

Sources:
    gym_trade/policy/trend/direction_toggle.py  (Policy + features)
    gym_trade/tool/ta.py::direction_toggle       (underlying signal — unchanged)

The previous implementation depended on `gym_trade.env.embodied.gym_trade.GymTradeEnv`,
which was deleted from the repo. Original code is available in git history.
"""
