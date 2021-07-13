from CustomNeuralNetwork import CustomNeuralNetwork
from utils import set_random_seed, wandb_log
import numpy as np
import torch
from memory_buffer import ReplayBuffer, ReplayBufferSamples
import wandb
from typing import Any, Dict, Optional

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
        target_noise_clip: Optional[float] = 0.5
    ):
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.previous_action = None
        self.previous_obs = None
        self.seed = None
        # neural network parameters
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.update_target_rate = update_target_rate
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        self.discount_factor = discount_factor
        # memory parameters
        self.replay_buffer = None
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.tot_timestep = 0
        self.init_actor(policy_estimator_info)
        self.init_critic(function_approximator_info)
        self.init_memory_buffer(memory_info)
        
        self.init_seed(seed)

    # ====== Initialization functions ==================================
    
    def init_actor(self, params):
        self.actor = CustomNeuralNetwork(**params)
        self.actor_target = CustomNeuralNetwork(**params)
    
    def init_critic(self, params):
        self.critic = CustomNeuralNetwork(**params)
        self.critic_target = CustomNeuralNetwork(**params)
    
    def init_memory_buffer(self, params):
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        self.replay_buffer = ReplayBuffer(**params)

    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.function_approximator.set_seed(seed)

    # ====== Action choice related functions ===========================

    def choose_action(self, obs:torch.Tensor):
        action = self.actor(obs)
        action = torch.clamp(action, -1, 1)
        return action


    # ====== Agent core functions ======================================

    def start(self, obs):
        current_action = self.actor(obs)
        self.previous_action = current_action
        self.previous_obs = obs

        return current_action

    def step(self, obs, reward):
        # storing the transition in the function approximator memory for further use
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
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self._update_target_net()
        # we can start learning when the memory is full
        if self.replay_buffer.full:
            # getting batch data
            batch = self.replay_buffer.sample()

            # value of the action being taken at the current timestep
            batch_oa = self._concat_obs_action(batch.observations, batch.actions)
            q_eval = self.critic(batch_oa)
            # values of the actions at the next step
            q_next = self.critic_target(batch_oa)
            q_next = self._zero_terminal_states(q_next, batch.next_observations)
            noise = self._create_noise_tensor(batch.actions)
            q_next += noise
            # Q containing only the max value of q in next step
            q_target = batch.rewards + self.discount_factor * q_next
            # computing the loss
            critic_loss = self.loss_func(q_eval, q_target.detach())

            self.critic.backpropagate(critic_loss)
            
            actions = self.actor(batch.observations)
            batch_oa = self._concat_obs_action(batch.observations, actions)
            actor_loss = - self.critic(batch_oa).mean()
            self.actor.backpropagate(actor_loss)

            # residual variance for plotting purposes (not sure if it is correct)
            #q_res = self.target_net(batch.observations).gather(1, batch.actions.long())
            #res_var = torch.var(q_res - q_eval) / torch.var(q_res)
            wandb_log({
                "Agent info/critic loss": critic_loss,
                "Agent info/actor loss": actor_loss
            })


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
        state = torch.cat((state, state)).unsqueeze(1)
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
        self.replay_buffer.reinit(state_dim, action_dim)