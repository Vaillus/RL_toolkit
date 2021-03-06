from CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import torch
from torch.distributions import Categorical
import wandb
from typing import Optional


MSELoss = torch.nn.MSELoss()

class PPOAgent:
    def __init__(
        self, 
        params
    ):
        self.γ = None
        self.state_dim = None
        self.num_actions = None

        self.policy_estimator: CustomNeuralNetwork = None
        self.function_approximator: CustomNeuralNetwork = None

        self.previous_state = None
        self.previous_action = None

        self.memory_size = None
        self.memory = []
        self.mem_cnt = 0

        self.seed = None

        self.tot_timestep = 0

        self.clipping = None
        self.value_coeff = None
        self.entropy_coeff = None
        self.n_epochs = None

        self.set_params_from_dict(params)
        self.set_other_params()

        # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.γ = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        
        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_function_approximator(params.get(
            "function_approximator_info"))
        wandb.watch(self.policy_estimator, log_freq=100)
        wandb.watch(self.function_approximator, log_freq=100)

        self.memory_size = params.get("memory_size", 200)
        self.state_dim = params.get("state_dim", 4)
        
        self.seed = params.get("seed", None)

        self.clipping = params.get("clipping", 0.2)
        self.value_coeff = params.get("value_coeff", 1.0)
        self.entropy_coeff = params.get("entropy_coeff", 0.01)
        self.n_epochs = params.get("n_epochs", 8)

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.init_memory()
     
    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    def initialize_function_approximator(self, params):
        self.function_approximator = CustomNeuralNetwork(params)
        #self.function_approximator_eval = CustomNeuralNetwork(params)
        #self.function_approximator_target = CustomNeuralNetwork(params)
    
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
        batch_state = torch.Tensor(self.memory[:, :self.state_dim]).float()
        batch_action = torch.Tensor(self.memory[:, 
            self.state_dim:self.state_dim + 1].astype(int)).float()
        batch_reward = torch.Tensor(self.memory[:, 
            self.state_dim + 1:self.state_dim + 2]).float()
        batch_next_state = torch.Tensor(self.memory[:, -self.state_dim-1:-1]).float()
        batch_is_terminal = torch.Tensor(self.memory[:, -1:]).bool()

        return batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal

    def init_memory(self):
         self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 3))
        
    def reset_memory(self):
        self.init_memory()
        self.mem_cnt = 0


    def control(self):  
        if self.can_learn():
            # initializing the memory related variables
            self.mem_cnt = 0
            batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal = self.sample_memory()
            # get discounted rewards
            batch_discounted_reward = torch.tensor(np.zeros((self.memory_size, 1))).float()
            disc_reward = 0.0
            for i, reward_i in enumerate(torch.flip(batch_reward, (0,1))):
                if batch_is_terminal[self.memory_size - 1 - i, 0]:
                    disc_reward = 0.0
                disc_reward = reward_i + self.γ * disc_reward
                batch_discounted_reward[self.memory_size - 1 - i, 0] = disc_reward
            
            # computing state values, advantage
            prev_state_value = self.function_approximator(batch_state)
            advantage = batch_discounted_reward - prev_state_value.detach()
            self.normalize(advantage) # is it really a good idea?
            # get probabilities of actions from policy estimator
            probs_old = self.policy_estimator(batch_state).detach()

            for epoch in range(self.n_epochs):
                probs_new = self.policy_estimator(batch_state)
                #ratio = probs_new / probs_old
                # in stable baselines 3, they write it this way but I 
                # don't know why.
                ratio = torch.exp(torch.log(probs_new) - torch.log(probs_old))
                ratio = torch.gather(ratio, 1, batch_action.long())
                clipped_ratio = torch.clamp(ratio, min = 1 - self.clipping, max = 1 + self.clipping) # OK
                policy_loss = torch.min(advantage.detach() * ratio, advantage.detach() * clipped_ratio) # OK
                policy_loss = - policy_loss.mean() # OK
                
                entropy = -(torch.sum(probs_new * torch.log(probs_new), dim=1, keepdim=True).mean())
                entropy_loss = entropy * self.entropy_coeff
                self.policy_estimator.backpropagate(policy_loss + entropy_loss)

                
                # TODO: clip the state value variation. nb: only openai does that.
                # delta_state_value = self.function_approximator_eval(batch_state) - prev_state_value
                # new_prev_state_value = prev_state_value + delta_state_value
                # state_value_error = 
                prev_state_value = self.function_approximator(batch_state)
                value_loss = MSELoss(prev_state_value, batch_discounted_reward)
                value_loss *= self.value_coeff
                self.function_approximator.backpropagate(value_loss)
                
                wandb.log({
                    'Agent info/critic loss': value_loss,
                    'Agent info/actor loss': policy_loss,
                    'Agent info/policy entropy': entropy})

                self.reset_memory()
                
    def normalize(self, tensor):
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor

    # ====== Action choice related functions ===========================

    def choose_action(self, state) -> int:
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

    def can_learn(self) -> bool:
        if self.mem_cnt == self.memory_size:
            return True
        else:
            return False

    def get_state_value_eval(self, state):
        if self.num_actions > 1:
            state_value = self.policy_estimator(state).data
        else: 
            state_value = self.function_approximator(state).data
        return state_value
        

