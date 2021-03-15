import numpy as np

class ProbeEnv:
    def __init__(self, name: str):
        self.name = name

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
    
    def reset(self):
        if self.name == "one":
            state_data = [0]
        if self.name == "two":
            obs = np.random.choice([-1, 1])
            state_data = [obs]
        return state_data
    
    def step(self, actions_data):
        if self.name == "one":
            new_state_data = [0]
            done = True
            reward_data = 1.0
        elif self.name == "two":
            obs = np.random.choice([-1, 1])
            new_state_data = [obs]
            done = True
            reward_data = obs
        other_data = None
        return new_state_data, reward_data, done, other_data

    def close(self):
        pass
