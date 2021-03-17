import numpy as np

class ProbeEnv:
    def __init__(self, name: str, writer):
        self.name = name
        self.witer = writer
        self.env2_value = None

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
        
    
    def reset(self):
        if self.name == "one":
            state_data = [0]
        if self.name == "two":
            obs = np.random.choice([-1, 1])
            self.env2_value = obs
            state_data = [obs]
        return state_data
    
    def step(self, actions_data):
        if self.name == "one":
            new_state_data = [0]
            done = True
            reward_data = 1.0
        elif self.name == "two":
            obs = self.env2_value
            new_state_data = [obs]
            done = True
            reward_data = obs
        other_data = None
        return new_state_data, reward_data, done, other_data

    def close(self):
        pass
