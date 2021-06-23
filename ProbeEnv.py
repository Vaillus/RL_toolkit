import numpy as np
import torch

#class ProbeEnv(ABC):

class ProbeEnv:
    def __init__(self, action_type: str, name: str, writer):
        self.action_type = action_type # discrete or continuous
        self.name = name
        self.writer = writer
        self.init_obs = None
        self.is_first_step = True

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
    
    
    def reset(self):
        if self.action_type == "discrete":
            self.reset_discrete()
        else:
            self.reset_continuous()

    def reset_discrete(self):
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
    
    def reset_continuous(self):
        if self.names == "one":
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
        if self.action_type == "discrete":
            self.step_discrete(actions_data)
        else:
            self.step_continuous(actions_data)

    def step_discrete(self, actions_data):
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
    
    def step_continuous(self, actions_data):
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
            if actions_data > 0.5:
                reward_data = -1.0
            if actions_data  <= 0.5:
                reward_data = 1.0
        elif self.name == "five":
            new_state_data = [0]
            done = True
            if self.init_obs == -1:
                if actions_data > 0.5:
                    reward_data = -1.0
                if actions_data  <= 0.5:
                    reward_data = 1.0
            elif self.init_obs == 1:
                if actions_data <= 0.5:
                    reward_data = -1.0
                if actions_data  > 0.5:
                    reward_data = 1.0
        
        other_data = None
        return new_state_data, reward_data, done, other_data


    def close(self):
        pass
    
    def plot_probe_envs(self, episode_id, agent):
        if self.action_type == "discrete":
            self.plot_probe_envs_discrete(episode_id, agent)
        else:
            self.plot_probe_envs_continuous(episode_id, agent)

    def plot_probe_envs_discrete(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_state_value_eval(state)
            self.writer.add_scalar("Probe/Value of state 0", state_value, episode_id)
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            self.writer.add_scalar("Probe/Value of state -1", state_value_neg, episode_id)
            self.writer.add_scalar("Probe/Value of state 1", state_value_pos, episode_id)
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.writer.add_scalar("Probe/Value of state 0", first_state_value, episode_id)
            self.writer.add_scalar("Probe/Value of state 1", second_state_value, episode_id)
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_state_value_eval(state)
            self.writer.add_scalar("Probe/state 0 action 0", actions_values[0], episode_id)
            self.writer.add_scalar("Probe/state 0 action 1", actions_values[1], episode_id)
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.writer.add_scalar("Probe/state -1 action 0", first_state_value[0], episode_id)
            self.writer.add_scalar("Probe/state -1 action 1", first_state_value[1], episode_id)
            self.writer.add_scalar("Probe/state 1 action 0", second_state_value[0], episode_id)
            self.writer.add_scalar("Probe/state 1 action 1", second_state_value[1], episode_id)

    def plot_probe_envs_continuous(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_state_value_eval(state)
            self.writer.add_scalar("Probe/Value of state 0", state_value, episode_id)
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            self.writer.add_scalar("Probe/Value of state -1", state_value_neg, episode_id)
            self.writer.add_scalar("Probe/Value of state 1", state_value_pos, episode_id)
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.writer.add_scalar("Probe/Value of state 0", first_state_value, episode_id)
            self.writer.add_scalar("Probe/Value of state 1", second_state_value, episode_id)
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_state_value_eval(state)
            self.writer.add_scalar("Probe/state 0 action 0", actions_values[0], episode_id)
            self.writer.add_scalar("Probe/state 0 action 1", actions_values[1], episode_id)
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.writer.add_scalar("Probe/state -1 action 0", first_state_value[0], episode_id)
            self.writer.add_scalar("Probe/state -1 action 1", first_state_value[1], episode_id)
            self.writer.add_scalar("Probe/state 1 action 0", second_state_value[0], episode_id)
            self.writer.add_scalar("Probe/state 1 action 1", second_state_value[1], episode_id)

    def show_probe_env_result(self, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_state_value_eval(state)
            print(f'value of state {state.data}: {state_value}')
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            print(f'value of state {state_pos.data}: {state_value_pos}')
            print(f'value of state {state_neg.data}: {state_value_neg}')
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            print(f'value of state {first_state.data}: {first_state_value}')
            print(f'value of state {second_state.data}: {second_state_value}')
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_state_value_eval(state)
            print(f'value of actions in state {state.data}: {actions_values}')
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_actions_values = agent.get_state_value_eval(first_state)
            second_state_actions_values = agent.get_state_value_eval(second_state)
            print(f'value of state {first_state.data}: {first_state_actions_values}')
            print(f'value of state {second_state.data}: {second_state_actions_values}')