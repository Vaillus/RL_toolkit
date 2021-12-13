from CustomNeuralNetwork import CustomNeuralNetwork
import torch
from torch.distributions import Categorical
from typing import Optional, Type, Dict, Any
from modules.replay_buffer import PPOReplayBuffer
from modules.logger import Logger
from utils import set_random_seed

from modules.curiosity import Curiosity
import numpy as np


MSELoss = torch.nn.MSELoss()

class PPOAgent:
    def __init__(
        self, 
        policy_estimator_info: Dict[str, Any],
        function_approximator_info: Dict[str, Any],
        memory_info: Dict[str, Any],
        curiosity_info: Optional[Dict[str, Any]] = None,
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

        self.curiosity = self.init_curiosity(curiosity_info)

        self.seed = seed
        self.clipping = clipping
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs

        self.previous_state = None
        self.previous_action = None

        self.memory_size = memory_info.get("size", 200) #why?

    # ====== Initialization functions ==================================
     
    def initialize_policy_estimator(self, params: Dict) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(**params, input_dim=self.state_dim, output_dim=self.num_actions)

    def initialize_function_approximator(self, params: Dict) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(**params, input_dim=self.state_dim, output_dim=self.num_actions)
    
    def init_memory_buffer(self, params: Dict) -> PPOReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        params["discount_factor"] = self.γ
        return PPOReplayBuffer(**params)

    def init_curiosity(self, params: Dict) -> Curiosity:
        if params:
            return Curiosity(**params)
        else:
            return Curiosity()

    
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
        self.logger.wandb_watch([self.actor, self.critic], type="grad")
    

    def control(self):
        if self.replay_buffer.full:
            batch = self.replay_buffer.sample()
            
            # computing state values, advantage
            prev_state_value = self.critic(batch.observations)

            advantage = batch.returns - prev_state_value.detach()
            self.normalize(advantage) # is it really a good idea?
            # get probabilities of actions from policy estimator
            probs_old = self.actor(batch.observations).detach()

            # initializing the lists containing the metrics to be logged
            value_losses = []
            policy_losses = []
            entropy_losses = []
            entropies = []
            intrinsic_rewards = []

            for _ in range(self.n_epochs):
                probs_new = self.actor(batch.observations)
                #ratio = probs_new / probs_old
                # in stable baselines 3, they write it this way but I 
                # don't know why.
                ratio = torch.exp(torch.log(probs_new) - torch.log(probs_old)) # ok
                ratio = torch.gather(ratio, 1, batch.actions.long()) #ok 
                clipped_ratio = torch.clamp(ratio, min = 1 - self.clipping, max = 1 + self.clipping) # OK
                policy_loss = torch.min(advantage.detach() * ratio, advantage.detach() * clipped_ratio) # OK
                policy_loss = - policy_loss.mean() # OK
                policy_losses.append(policy_loss.item())
                
                entropy = -(torch.sum(probs_new * torch.log(probs_new), dim=1, keepdim=True).mean())
                entropies.append(entropy.item())
                entropy_loss = - entropy * self.entropy_coeff
                entropy_losses.append(entropy_loss.item())

                # computing ICM loss
                intrinsic_reward = self.compute_icm_loss(batch, self.actor)
                intrinsic_rewards.append(intrinsic_reward.item())
                
                self.actor.backpropagate(policy_loss + entropy_loss + intrinsic_reward)

                # TODO: clip the state value variation. nb: only openai does that. nb2: It is not recommended anyway.
                # delta_state_value = self.function_approximator_eval(batch_state) - prev_state_value
                # new_prev_state_value = prev_state_value + delta_state_value
                # state_value_error = 
                prev_state_value = self.critic(batch.observations)
                value_loss = MSELoss(prev_state_value, batch.returns)
                value_losses.append(value_loss.item())
                value_loss *= self.value_coeff
                self.critic.backpropagate(value_loss)
                
            self.logger.wandb_log({
                'Agent info/critic loss': np.mean(value_losses),
                'Agent info/actor loss': np.mean(policy_losses),
                'Agent info/entropy': np.mean(entropies),
                'Agent info/entropy loss': np.mean(entropy_losses),
                'Agent info/intrinsic reward': np.mean(intrinsic_rewards)},
                type= "agent")
            # the replay buffer is used only one (completely) and then 
            # emptied out
            self.replay_buffer.erase()

    def compute_icm_loss(self, batch, actor):
        return self.curiosity.compute_icm_loss(batch=batch, nn=actor)
                
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
        intrinsic_reward = self.curiosity.get_intrinsic_reward(self.actor, self.previous_state, state, self.previous_action)
        self.replay_buffer.store_transition(self.previous_state, self.previous_action, reward + intrinsic_reward, state, False)
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.control()
        #self.vanilla_control(state, reward, False)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        intrinsic_reward = self.curiosity.get_intrinsic_reward(self.actor, self.previous_state, state, self.previous_action)
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(self.previous_state, self.previous_action, reward + intrinsic_reward, state, True)
        self.control()

    def get_state_value_eval(self, state):
        if self.num_actions > 1:
            state_value = self.actor(state).data
        else: 
            state_value = self.critic(state).data
        return state_value

    def adjust_dims(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.num_actions = action_dim
        self.actor.reinit_layers(state_dim, action_dim)
        self.critic.reinit_layers(state_dim, 1)
        self.replay_buffer.correct(state_dim, action_dim)
        self.logger.wandb_watch([self.actor, self.critic])

