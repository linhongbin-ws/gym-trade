from abc import ABC, abstractmethod
import gym
from gym import spaces
from gym.utils import seeding

class BaseEnv(gym.Env, ABC):
    """
    Base environment for Gym < 0.26
    """

    metadata = {"render.modes": []}

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._seed = None

    # ---------- Core Gym API ----------

    @abstractmethod
    def reset(self):
        """
        Returns:
            obs
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Returns:
            obs, reward, done, info
        """
        raise NotImplementedError

    # ---------- Spaces ----------

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError

    # ---------- Seeding (old Gym style) ----------

    def seed(self, seed=None):
        # Base seed
        self.env_rng, seed = seeding.np_random(seed)

        # Derive deterministic sub-seeds
        obs_seed = seed + 1
        action_seed = seed + 2

        self.obs_rng, _ = seeding.np_random(obs_seed)
        self.action_rng, _ = seeding.np_random(action_seed)

        return [seed, obs_seed, action_seed]

    # ---------- Optional helpers ----------

    @property
    def t(self):
        """Current timestep (optional)"""
        return None

    @property
    def unwrapped(self):
        return self

    @property
    def is_wrapper(self):
        return False
