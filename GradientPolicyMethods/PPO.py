from CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import torch
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self):
        self.γ = None
        self.state_dim = None
        self.num_actions = None

        self.policy_estimator = None
        self.function_approximator = None

        self.previous_state = None
        self.previous_action = None

        self.memory_size = None
        self.memory = []
        self.mem_cnt = 0

        self.seed = None

        self.writer = None
        self.tot_timestep = 0

        self.set_params_from_dict(params)
        self.set_other_params()

        # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.γ = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        
        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_function_approximator(params.get(
            "function_approximator_info"))

        self.memory_size = params.get("memory_size", 200)
        self.state_dim = params.get("state_dim", 4)
        
        self.seed = params.get("seed", None)

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 3))
     
    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    def initialize_function_approximator(self, params):
        self.function_approximator = CustomNeuralNetwork(params)
        #self.function_approximator_eval = CustomNeuralNetwork(params)
        #self.function_approximator_target = CustomNeuralNetwork(params)
    
    def init_memory(self):
         self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 3))
        
    def reset_memory(self):
        self.init_memory()
        self.mem_cnt = 0

    # ====== Memory functions ==========================================

    def store_transition(self, state, action, reward, next_state, is_terminal):
        # store a transition (SARS') in the memory
        is_terminal = [is_terminal]
        transition = np.hstack((state, [action, reward], next_state, is_terminal))
        self.memory[self.mem_cnt % self.memory_size, :] = transition
        self.incr_mem_cnt()
        
    def incr_mem_cnt(self):
        # increment the memory counter and resets it to 0 when reached 
        # the memory size value to avoid a too large value
        self.mem_cnt += 1

    def sample_memory(self):
        # Getting the batch of samples corresponding to those indices 
        # and dividing it into state, action, reward and next state
        batch_state = self.memory[:, :self.state_dim]).float()
        batch_action = self.memory[:, 
            self.state_dim:self.state_dim + 1].astype(int)).float()
        batch_reward = self.memory[:, 
            self.state_dim + 1:self.state_dim + 2]).float()
        batch_next_state = self.memory[:, -self.state_dim-1:-1]).float()
        batch_is_terminal = self.memory[:, -1:]).bool()

        return batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal

    def control(self):
        if (self.mem_cnt == self.memory_size) and self.can_learn():
            self.mem_cnt = 0
            batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal = self.sample_memory()
            prev_state_value = self.function_approximator_eval(batch_state)
            state_value = self.function_approximator_target(batch_next_state)
            nu_state_value = torch.zeros(state_value.shape)
            nu_state_value = torch.masked_fill(state_value, batch_is_terminal, 0.0)

            δ = reward + self.γ * nu_state_value.detach() - prev_state_value.detach()

            value_loss = - prev_state_value * δ
            value_loss = value_loss.mean()
            self.function_approximator_eval.optimizer.zero_grad()
            value_loss.backward()
            self.function_approximator_eval.optimizer.step()
            self.writer.add_scalar("Agent info/critic loss", value_loss, self.tot_timestep)

            # plot the policy entropy
            batch_probs = self.policy_estimator(batch_state).detach().numpy()
            entropy = -(np.sum(probs * np.log(probs)))
            self.writer.add_scalar("Agent info/policy entropy", entropy, self.tot_timestep)

            logprob = - torch.log(self.policy_estimator(self.previous_state)[self.previous_action])
            loss = logprob * δ 
            self.policy_estimator.optimizer.zero_grad()
            loss.backward()
            self.policy_estimator.optimizer.step()
            self.writer.add_scalar("Agent info/actor loss", loss, self.tot_timestep)
            self.empty_memory()
    
    def can_learn(self):
        if self.mem_cnt == self.memory_size:
            return True
        else:
            return False

    


 # ====== Action choice related functions ===========================

    def choose_action(self, state): # TODO fix first if
        action_probs = Categorical(self.policy_estimator(state))
        action_chosen = action_probs.sample()
        return action_chosen.item()

    # ====== Agent core functions ======================================

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):

        # storing the transition in the function approximator memory for further use
        self.store_transition(self.previous_state, self.previous_action, reward, state, False)
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.control()
        #self.vanilla_control(state, reward, False)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.store_transition(self.previous_state, self.previous_action, reward, state, True)
        self.control()
    