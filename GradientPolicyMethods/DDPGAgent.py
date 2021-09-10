from CustomNeuralNetwork import CustomNeuralNetwork
from utils import set_random_seed
import numpy as np
import torch
from replay_buffer import ReplayBuffer
from typing import Any, Dict, Optional, Type
from copy import deepcopy
from modules.logger import Logger

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:
    def __init__(
        self, 
        policy_estimator_info: Dict[str, Any],
        function_approximator_info: Dict[str, Any],
        memory_info: Dict[str, Any],
        seed: Optional[int] = 0,
        num_actions: Optional[int] = 1,
        state_dim: Optional[int] = 1,
        update_target_rate: Optional[int] = 50,
        discount_factor: Optional[float] = 0.995,
        target_policy_noise: Optional[float] = 0.2,
        target_noise_clip: Optional[float] = 0.5,
        her: Optional[bool] = False
    ):
        self.num_actions = num_actions
        self.state_dim = state_dim
        
        self.seed = self.init_seed(seed)
        self.logger = None
        # neural network parameters
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.update_target_rate = update_target_rate
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()

        self.γ = discount_factor
        self.her = her
        self.replay_buffer = self.init_memory_buffer(memory_info)
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.tot_timestep = 0
        self.init_actor(policy_estimator_info)
        self.init_critic(function_approximator_info)

        self.previous_action = None
        self.previous_obs = None
        
        

    # ====== Initialization functions ==================================
    
    def init_actor(self, params):
        self.actor = CustomNeuralNetwork(**params, input_dim=self.state_dim, output_dim=self.num_actions)
        self.actor_target = deepcopy(self.actor)
    
    def init_critic(self, params):
        self.critic = CustomNeuralNetwork(**params, input_dim=self.state_dim, output_dim=self.num_actions)
        self.critic_target = deepcopy(self.critic)
    
    def init_memory_buffer(self, params) -> ReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        return ReplayBuffer(**params)

    def init_seed(self, seed):
        if seed:
            set_random_seed(self.seed)
            return seed

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.actor.set_seed(seed)
            self.actor_target.set_seed(seed)
            self.critic.set_seed(seed)
            self.critic_target.set_seed(seed)
            # TODO: set memory buffer seed?
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch([self.actor, self.critic])

    def get_discount(self):
        return self.γ

    # ====== Action choice related functions ===========================

    def choose_action(self, obs:torch.Tensor):
        action = self.actor(obs)
        noise = np.random.normal(0,self.target_noise_clip)
        action += noise
        action = torch.clamp(action, -1, 1)
        return action


    # ====== Agent core functions ======================================

    def start(self, obs):
        current_action = self.choose_action(obs)
        self.previous_action = current_action
        self.previous_obs = obs

        return current_action

    def step(self, obs, reward):
        # storing the transition in the function approximator memory for further use
        # TODO: add HER option
        self.replay_buffer.store_transition(self.previous_obs, self.previous_action, 
                                            reward, obs, False)
        # getting the action values from the function approximator
        current_action = self.choose_action(obs)
        self.control()
        self.previous_action = current_action
        self.previous_obs = obs
        
        return current_action

    def end(self, obs, reward):
        self.replay_buffer.store_transition(self.previous_obs, 
                                    self.previous_action, reward, obs, True)
        self.control()

    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        action_value = self.critic(state)
        noise = 0 # normal distrib, for exploration
        action_value = self.critic(state) + noise
        action_value = torch.clamp(action_value, self.min_action, self.max_action)
        return action_value

    # === parameters update functions ==================================

    def _update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_counter += 1
        if self.update_target_counter == self.update_target_rate:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.update_target_counter = 0
        
    
    # ====== Control related functions =================================

    def control(self):
        self._learn()
    
    def _learn(self):
        if self.replay_buffer.full:
            self._update_target_net()
            # getting batch data
            batch = self.replay_buffer.sample()
            # compute critic target
            target_actions = self.actor_target(batch.next_observations).detach()
            batch_oa = self._concat_obs_action(batch.next_observations, target_actions)
            q_next = self.critic_target(batch_oa).detach()
            q_next = (1.0 - batch.dones.float()) * q_next
            y = batch.rewards + self.γ * q_next
            batch_oa_eval = self._concat_obs_action(batch.observations, batch.actions)
            # compute critic eval
            q_eval = self.critic(batch_oa_eval)
            # learn critic
            critic_loss = self.loss_func(q_eval, y)
            
            self.critic.backpropagate(critic_loss)

            actor_eval = self.actor(batch.observations)
            #with torch.no_grad():
            test_oa = self._concat_obs_action(batch.observations, actor_eval)
            actor_loss = self.critic(test_oa)
            actor_loss = - actor_loss.mean()
            self.logger.wandb_log({
                "critic loss": critic_loss,
                "actor loss": actor_loss
            })
            
            self.actor.backpropagate(actor_loss)



    def _concat_obs_action(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action), 1)#.unsqueeze(1)),1)
        return obs_action
    
    def get_action_value_eval(self, state:torch.Tensor):
        """for plotting purposes only?
        """
        action = np.random.uniform(-1, 1, 1)
        action = torch.Tensor(action)
        state_action = torch.cat((state, action))
        action_value = self.critic(state_action).detach().data
        return action_value
    
    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        """ for plotting purposes only?
        """
        #state = torch.cat((state, state)).unsqueeze(1)
        state = (state.unsqueeze(1) * torch.ones(len(actions))).T
        state_action = torch.cat((state, actions.unsqueeze(1)),1)
        action_values = self.critic(state_action).data
        return action_values
    
    def _zero_terminal_states(self,  q_values: torch.Tensor,
                                     dones:torch.Tensor) -> torch.Tensor:
        """ Zeroes the q values at terminal states
        """
        nu_q_values = torch.zeros(q_values.shape)
        nu_q_values = torch.masked_fill(q_values, dones, 0.0)
        return nu_q_values
    
    def _create_noise_tensor(self, tensor):
        # create the nois tensor filled with normal distribution
        noise = tensor.clone().data.normal_(0, self.target_policy_noise)
        # clip the normal distribution
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        return noise
    
    def adjust_dims(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.num_actions = action_dim
        self.actor.reinit_layers(state_dim, action_dim)
        self.actor_target.reinit_layers(state_dim, action_dim)
        self.critic.reinit_layers(state_dim + action_dim, 1)
        self.critic_target.reinit_layers(state_dim + action_dim, 1)
        self.replay_buffer.correct(state_dim, action_dim)
