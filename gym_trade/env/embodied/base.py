from abc import ABC, abstractmethod
class BaseEnv(ABC):
    def __init__(self, client):
        self.client = client
        
    @property
    def unwrapped(self):
        return self
    
    def __del__(self):
    # cv2.destroyAllWindows()
        del self.client
    
    @property
    def is_wrapper(self):
        return False
    

    @abstractmethod
    def reset(self):
        return self.client.reset()

    @abstractmethod
    def step(self,action):
        obs, reward, done, info = self.client.step(action)
        return obs, reward, done, info

    @abstractmethod
    def render(self, **kwargs): #['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)
    
    @abstractmethod
    def get_oracle_action(self,obs):
        return self.client.get_oracle_action(obs)


    @property
    @abstractmethod
    def action_space(self):
        return self.client.action_space

    
    @property
    @abstractmethod
    def observation_space(self):
        return self.client.observation_space

    
    @property
    @abstractmethod
    def seed(self):
        return self.client.seed

    
    @property
    @abstractmethod
    def timestep(self):
        return self.client.timestep

    
    @seed.setter
    @abstractmethod
    def seed(self, seed):
        self.client.seed = seed


        