from gym_trade.env.embodied.base import BaseEnv


class GymTradeEnv(BaseEnv):
    def __init__(self, task, 
                    **kwargs,
                    ):
        if task=="us_stock":
            from gym_trade.env.embodied.gym_trade.us_stock import US_Stock_Env
            client = US_Stock_Env(**kwargs)
        else:
            raise NotImplementedError
        super().__init__(client)

    def reset(self):
        return self.client.reset()

    def step(self,action):
        obs, reward, done, info = self.client.step(action)
        return obs, reward, done, info

    def load_stock_list(self, file_list):
        self.client._update_csv_dir(file_list)

    def render(self, **kwargs):
        return self.client.render()
    
    def get_oracle_action(self,obs):
        return self.client.get_oracle_action(obs)

    @property
    def action_space(self):
        return self.client.action_space

    @property
    def observation_space(self):
        return self.client.observation_space

    @property
    def seed(self):
        return self.client.seed

    @property
    def timestep(self):
        return self.client.timestep

    @seed.setter
    def seed(self, seed):
        self.client.seed = seed
    
    @property
    def df(self,):
        return self.client._df

    @property
    def file(self,):
        return self.client.file
    
    @property
    def pnl(self):
        return self.client.pnl