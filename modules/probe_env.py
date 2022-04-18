import numpy as np
import torch
from modules.logger import Logger
import matplotlib.pyplot as plt
#from typing import Type

#class ProbeEnv(ABC):

class ProbeEnv:
    def __init__(self, name: str):
        self.name = name
        self.init_obs = None
        self.is_first_step = True
        self.logger = None

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
    
    def set_logger(self, logger: Logger):
        self.logger = logger
    
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

    def step(self, action:int):
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
            if action == 0:
                reward_data = -1.0
            if action == 1:
                reward_data = 1.0
        elif self.name == "five":
            new_state_data = [0]
            done = True
            if self.init_obs == -1:
                if action == 0:
                    reward_data = 1.0
                if action == 1:
                    reward_data = -1.0
            elif self.init_obs == 1:
                if action == 0:
                    reward_data = -1.0
                if action == 1:
                    reward_data = 1.0

        other_data = None
        return new_state_data, reward_data, done, other_data

    def plot(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            state_value = agent.get_state_value_eval(state)
            self.logger.log({"Probe/Value of action at state 0": state_value}, 1)
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            self.logger.log({
                "Probe/Value of state -1": state_value_neg,
                "Probe/Value of state 1": state_value_pos
            },1)
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.logger.log({
                "Probe/Value of state 0": first_state_value,
                "Probe/Value of state 1": second_state_value
            },1)
        elif self.name == "four":
            state = torch.tensor([0])
            actions_values = agent.get_state_value_eval(state)
            self.logger.log({
                "Probe/state 0 action 0": actions_values[0],
                "Probe/state 0 action 1": actions_values[1]
            },1)
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            self.logger.log({
                'Probe/state -1 action 0': first_state_value[0],
                'Probe/state -1 action 1': first_state_value[1],
                'Probe/state 1 action 0': second_state_value[0],
                'Probe/state 1 action 1': second_state_value[1]},1)
            
    
    def show_results(self, agent):
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
    
    def step(self, action:float):
        self.logger.log({"Probe/Action value": action}, 100)
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
                obs = 0
                new_state_data = [obs]
                done = True
                reward_data = 1.0
        elif self.name == "four":
            new_state_data = [0]
            done = True
            reward_data = ContinuousProbeEnv.gaussian(action, 0, 1)
            
        elif self.name == "five":
            obs = 0
            new_state_data = [obs]
            done = True
            reward_data = ContinuousProbeEnv.gaussian(action, self.init_obs * 0.5, 0.5)
        
        other_data = None
        return new_state_data, reward_data, done, other_data
    

    def plot(self, episode_id, agent):
        if self.name == "one":
            state = torch.tensor([0])
            action = torch.tensor([0])
            #action_value = agent.get_action_values_eval(state, action).item()
            state_value = agent.get_state_value_eval(state)
            self.logger.log({"Probe/Value of state 0": state_value},1)
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            #state_value_pos = agent.get_action_values_eval(state_pos, torch.tensor([0.0]))
            #state_value_neg = agent.get_action_values_eval(state_neg, torch.tensor([0.0]))
            self.logger.log({
                "Probe/Value of state -1": state_value_neg,
                "Probe/Value of state 1": state_value_pos
            },1)
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            #state_value_pos = agent.get_action_values_eval(first_state, torch.tensor([0.0]))
            #state_value_neg = agent.get_action_values_eval(second_state, torch.tensor([0.0]))
            self.logger.log({
                "Probe/Value of first state": first_state_value,
                "Probe/Value of second state": second_state_value
            },1)
        elif self.name == "four":
            pass
            #self.plot_final_result(agent, 0)
            state = torch.tensor([0])
            mu, sig = agent.actor(state)
            """action = torch.tensor([0])
            action_value = agent.get_action_values_eval(state, action).item()
            self.logger.log({"Probe/Value of action 0 at state 0": action_value}, 1)"""
            self.logger.log({"Probe/Mean action": mu, "Probe/std action": sig}, 1)
        elif self.name == "five":
            first_state = torch.tensor([-1])
            second_state = torch.tensor([1])
            mu1, sig1 = agent.actor(first_state)
            mu2, sig2 = agent.actor(second_state)
            self.logger.log({
                "Probe/Mean action for state -1": mu1, 
                "Probe/std action for state -1": sig1
            }, 1)
            self.logger.log({
                "Probe/Mean action for state +1": mu2, 
                "Probe/std action for state +1": sig2
            }, 1)
            #self.plot_final_result(agent, -1, -0.5, 0.5)
            #self.plot_final_result(agent, 1, 0.5, 0.5)
            """state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_action_values_eval(state_pos, torch.tensor([0.5]))
            state_value_neg = agent.get_action_values_eval(state_neg, torch.tensor([-0.5]))
            self.logger.log({
                "Probe/Value of action -0.5 at  state -1": state_value_neg,
                "Probe/Value of action 0.5 at state 1": state_value_pos
            }, 1)"""

    def show_results(self, agent):
        if self.name == "one":
            state = torch.tensor([0])
            action = torch.tensor([0])
            #action_value = agent.get_action_values_eval(state, action).item()
            state_value = agent.get_state_value_eval(state)
            #self.logger.log({"Probe/Value of state 0": state_value},1)
            print(f'value of state {state.data}: {state_value}')

            self.plot_final_result(agent, 0, mu= 1, mode="constant")
        elif self.name == "two":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            print(f'value of state {state_pos.data}: {state_value_pos}')
            print(f'value of state {state_neg.data}: {state_value_neg}')
            self.plot_final_result(agent, -1,  mu= -1, mode="constant")
            self.plot_final_result(agent, 1,  mu= 1, mode="constant")
        elif self.name == "three":
            first_state = torch.tensor([0])
            second_state = torch.tensor([1])
            first_state_value = agent.get_state_value_eval(first_state)
            second_state_value = agent.get_state_value_eval(second_state)
            print(f'value of state {first_state.data}: {first_state_value}')
            print(f'value of state {second_state.data}: {second_state_value}')
            self.plot_final_result(agent, 0, agent.get_discount() , mode="constant")
            self.plot_final_result(agent, 1, 1, mode="constant")
        elif self.name == "four":
            state = torch.tensor([0])
            action_value = agent.get_state_value_eval(state).item()
            print(f'Value of action 0 at state 0: {action_value}')
            self.plot_final_result(agent, 0)
        elif self.name == "five":
            state_pos = torch.tensor([1])
            state_neg = torch.tensor([-1])
            state_value_pos = agent.get_state_value_eval(state_pos)
            state_value_neg = agent.get_state_value_eval(state_neg)
            print(f'value of state {state_pos.data}: {state_value_pos}')
            print(f'value of state {state_neg.data}: {state_value_neg}')
            self.plot_final_result(agent, -1, -0.5, 0.5)
            self.plot_final_result(agent, 1, 0.5, 0.5)
    
    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def plot_final_result(self, agent, state_value, mu=0, sigma=1, mode="gaussian"):
        """ function that creates a plot to make sure the agent predicts the
        action values correctly.
        A first line (red) represents the reward of the actions, taking
        the shape of a gaussian, and the other line (blue) displays the 
        action value predictions of the agent. If everything works correctly,
        the two line should overlap.
        """
        x = np.arange(-1,1, 0.1)
        y = agent.get_action_values_eval(
            torch.tensor([state_value]), torch.tensor(x))
        if mode == "gaussian":
            baseline = ContinuousProbeEnv.gaussian(x, mu, sigma)
        elif mode == "constant":
            baseline = np.ones(x.shape) * mu
        plt.plot(x, y, label='output of critic')
        plt.plot(x, baseline, 'r', label='reward')
        plt.xlabel("Action value")
        plt.legend()
        self.logger.wandb_plot({f"Probe/actions values at state {state_value}": plt})

class PerfoProbeEnv(ProbeEnv):
    def __init__(self, name: str):
        super(DiscreteProbeEnv, self).__init__(name)
    
    def reset(self):
        if self.name == "one":
            state_data = [0]
        elif self.name == "two":
            obs = np.random.choice([-1, 1])
            self.init_obs = obs
            state_data = [obs]
        return state_data
