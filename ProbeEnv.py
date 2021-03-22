import numpy as np

class ProbeEnv:
    def __init__(self, name: str, writer):
        self.name = name
        self.writer = writer
        self.init_obs = None
        self.is_first_step = True

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
        
    
    def reset(self):
        if self.name == "one":
            state_data = [0]
        elif self.name == "two":
            obs = np.random.choice([-1, 1])
            self.init_obs = obs
            state_data = [obs]
        elif self.name == "three":
            state_data = [0]
            self.is_first_step = True
        elif self.name == "four":
            state_data = [0]
        elif self.name == "five":
            obs = np.random.choice([-1, 1])
            self.init_obs = obs
            state_data = [obs]
        return state_data
    
    def step(self, actions_data):
        if self.name == "one":
            new_state_data = [0]
            done = True
            reward_data = 1.0
        elif self.name == "two":
            obs = 0
            new_state_data = [obs]
            done = True
            reward_data = self.init_obs
        elif self.name == "three":
            if self.is_first_step:
                obs = 1
                new_state_data = [obs]
                done = False
                reward_data = 0.0
                self.is_first_step = False
            else:
                obs = -1
                new_state_data = [obs]
                done = True
                reward_data = 1.0
        elif self.name == "four":
            new_state_data = [0]
            done = True
            if actions_data == 0:
                reward_data = -1.0
            if actions_data == 1:
                reward_data = 1.0
        elif self.name == "five":
            new_state_data = [0]
            done = True
            if self.init_obs == -1:
                if actions_data == 0:
                    reward_data = 1.0
                if actions_data == 1:
                    reward_data = -1.0

            elif self.init_obs == 1:
                if actions_data == 0:
                    reward_data = -1.0
                if actions_data == 1:
                    reward_data = 1.0

        other_data = None
        return new_state_data, reward_data, done, other_data

    def close(self):
        pass
