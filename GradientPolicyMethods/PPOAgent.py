from CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import torch
from torch.distributions import Categorical
import wandb
from typing import Optional, Type, Dict, Any
from replay_buffer import ReplayBuffer
from logger import Logger
from utils import set_random_seed


MSELoss = torch.nn.MSELoss()

class PPOAgent:
    def __init__(
        self, 
        policy_estimator_info: Dict[str, Any],
        function_approximator_info: Dict[str, Any],
        memory_info: Dict[str, Any],
        seed: Optional[int] = 0,
        discount_factor: Optional[float] = 0.9,
        num_actions: Optional[int] = 1,
        state_dim: Optional[int] = 1, 
        clipping: Optional[float] = 0.2,
        value_coeff: Optional[float] = 1.0,
        entropy_coeff: Optional[float] = 0.01,
        n_epochs: Optional[int] = 8
    ):
        self.γ = discount_factor
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.actor: CustomNeuralNetwork =\
             self.initialize_policy_estimator(policy_estimator_info)
        self.critic: CustomNeuralNetwork =\
             self.initialize_function_approximator(function_approximator_info)
        self.replay_buffer = self.init_memory_buffer(memory_info)

        self.seed = seed
        self.clipping = clipping
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs

        self.previous_state = None
        self.previous_action = None

     # ====== Initialization functions ==================================
     
    def initialize_policy_estimator(self, params: Dict) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(**params)

    def initialize_function_approximator(self, params: Dict) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(**params)
    
    def init_memory_buffer(self, params: Dict) -> ReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        return ReplayBuffer(**params)
    
    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.critic.set_seed(seed)
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch([self.actor, self.critic])
    

    def control(self):  
        if self.replay_buffer.full:
            batch = self.replay_buffer.sample()
            # get discounted rewards
            batch_discounted_reward = torch.tensor(np.zeros((self.replay_buffer.batch_size, 1))).float()
            disc_reward = 0.0
            for i, reward_i in enumerate(torch.flip(batch.rewards, (0,1))):
                if batch.dones[self.replay_buffer.batch_size - 1 - i, 0]:
                    disc_reward = 0.0
                disc_reward = reward_i + self.γ * disc_reward
                batch_discounted_reward[self.replay_buffer.batch_size - 1 - i, 0] = disc_reward
            
            # computing state values, advantage
            prev_state_value = self.critic(batch.observations)
            advantage = batch_discounted_reward - prev_state_value.detach()
            self.normalize(advantage) # is it really a good idea?
            # get probabilities of actions from policy estimator
            probs_old = self.actor(batch.observations).detach()

            for _ in range(self.n_epochs):
                probs_new = self.actor(batch.observations)
                #ratio = probs_new / probs_old
                # in stable baselines 3, they write it this way but I 
                # don't know why.
                ratio = torch.exp(torch.log(probs_new) - torch.log(probs_old))
                ratio = torch.gather(ratio, 1, batch.actions.long())
                clipped_ratio = torch.clamp(ratio, min = 1 - self.clipping, max = 1 + self.clipping) # OK
                policy_loss = torch.min(advantage.detach() * ratio, advantage.detach() * clipped_ratio) # OK
                policy_loss = - policy_loss.mean() # OK
                
                entropy = -(torch.sum(probs_new * torch.log(probs_new), dim=1, keepdim=True).mean())
                entropy_loss = entropy * self.entropy_coeff
                self.actor.backpropagate(policy_loss + entropy_loss)

                
                # TODO: clip the state value variation. nb: only openai does that.
                # delta_state_value = self.function_approximator_eval(batch_state) - prev_state_value
                # new_prev_state_value = prev_state_value + delta_state_value
                # state_value_error = 
                prev_state_value = self.critic(batch.observations)
                value_loss = MSELoss(prev_state_value, batch_discounted_reward)
                value_loss *= self.value_coeff
                self.critic.backpropagate(value_loss)
                
                self.logger.wandb_log({
                    'Agent info/critic loss': value_loss,
                    'Agent info/actor loss': policy_loss,
                    'Agent info/policy entropy': entropy})

                
    def normalize(self, tensor):
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor

    # ====== Action choice related functions ===========================

    def choose_action(self, state) -> int:
        action_probs = Categorical(self.actor(state))
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
        self.replay_buffer.store_transition(self.previous_state, self.previous_action, reward, state, False)
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.control()
        #self.vanilla_control(state, reward, False)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(self.previous_state, self.previous_action, reward, state, True)
        self.control()

    def get_state_value_eval(self, state):
        if self.num_actions > 1:
            state_value = self.actor(state).data
        else: 
            state_value = self.critic(state).data
        return state_value
        

