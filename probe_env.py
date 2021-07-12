import numpy as np
import torch
import wandb

#class ProbeEnv(ABC):

class ProbeEnv:
    def __init__(self, name: str):
        self.name = name
        self.init_obs = None
        self.is_first_step = True

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
    
    def close(self):
        pass

class DiscreteProbeEnv(ProbeEnv):
    def __init__(self, name: str):
        super(DiscreteProbeEnv, self).__init__(name)
    
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

    def plot(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_state_value_eval(state)
            wandb.log({"Probe/Value of action at state 0": state_value})
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            wandb.log({
                "Probe/Value of state -1": state_value_neg,
                "Probe/Value of state 1": state_value_pos
            })
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            wandb.log({
                "Probe/Value of state 0": first_state_value,
                "Probe/Value of state 1": second_state_value
            })
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_state_value_eval(state)
            wandb.log({
                "Probe/state 0 action 0": actions_values[0],
                "Probe/state 0 action 1": actions_values[1]
            })
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            wandb.log({
                'Probe/state -1 action 0': first_state_value[0],
                'Probe/state -1 action 1': first_state_value[1],
                'Probe/state 1 action 0': second_state_value[0],
                'Probe/state 1 action 1': second_state_value[1]})
            
    
    def show_result(self, agent):
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



class ContinuousProbeEnv(ProbeEnv):
    def __init__(self, name: str):
        super(ContinuousProbeEnv, self).__init__(name)
    
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
    

    def plot(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_action_value_eval(state)
            wandb.log({"Probe/Value of state 0": state_value})
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_action_value_eval(state_pos)
            state_value_neg = agent.get_action_value_eval(state_neg)
            wandb.log({
                "Probe/Value of state -1": state_value_neg,
                "Probe/Value of state 1": state_value_pos
            })
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_action_value_eval(first_state)
            second_state_value = agent.get_action_value_eval(second_state)
            wandb.log({
                "Probe/Value of state 0": first_state_value,
                "Probe/Value of state 1": second_state_value
            })
        elif self.name == "four":
            state = torch.tensor([0])
            actions = torch.tensor([0.25, 0.75])
            actions_values = agent.get_action_values_eval(state, actions)
            wandb.log({
                "Probe/state 0 action 0": actions_values[0],
                "Probe/state 0 action 1": actions_values[1]
            })
            
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            actions = torch.tensor([0.25, 0.75])
            first_state_av = agent.get_action_values_eval(first_state, actions)
            second_state_av = agent.get_action_values_eval(second_state, actions)
            wandb.log({
                "Probe/state -1 action 0.25": first_state_av[0],
                'Probe/state -1 action 0.75': first_state_av[1],
                'Probe/state 1 action 0.25': second_state_av[0],
                'Probe/state 1 action 0.75': second_state_av[1]})

    def show_results(self, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_action_value_eval(state)
            print(f'value of state {state.data}: {state_value}')
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_action_value_eval(state_pos)
            state_value_neg = agent.get_action_value_eval(state_neg)
            print(f'value of state {state_pos.data}: {state_value_pos}')
            print(f'value of state {state_neg.data}: {state_value_neg}')
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_action_value_eval(first_state)
            second_state_value = agent.get_action_value_eval(second_state)
            print(f'value of state {first_state.data}: {first_state_value}')
            print(f'value of state {second_state.data}: {second_state_value}')
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_action_value_eval(state)
            print(f'value of actions in state {state.data}: {actions_values}')
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_av = agent.get_action_values_eval(first_state, [0.25, 0.75])
            second_state_av = agent.get_state_value_eval(second_state, [0.25, 0.75])
            print(f'values of state {first_state.data}: {first_state_av}')
            print(f'values of state {second_state.data}: {second_state_av}')