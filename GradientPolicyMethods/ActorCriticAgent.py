from CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import torch
from torch.distributions import Categorical

from modules.replay_buffer import VanillaReplayBuffer
from typing import Optional, Type, Dict, Any
from modules.logger import Logger
from utils import set_random_seed
from scipy.stats import norm
import math

class ActorCriticAgent:
    def __init__(
        self, 
        discount_factor: float = 0.9,
        num_actions: int = 1,
        is_continuous: bool = False,
        policy_estimator_info: dict = {},
        function_approximator_info: dict = {},
        memory_info: dict = {},
        update_target_rate: int = 50,
        state_dim: int = 4,
        seed: int = None,
        is_vanilla: bool = True,
    ):
        # parameters to be set from params dict
        self.γ = discount_factor
        self.num_actions = num_actions
        self.is_continuous = is_continuous

        self.actor = self.initialize_policy_estimator(
            policy_estimator_info)

        self.critic_eval = self.initialize_function_approximator(
            function_approximator_info)
        self.critic_target = self.initialize_function_approximator(
            function_approximator_info)

        self.previous_state = None
        self.previous_action = None
        #self.rewards = []
        

        self.update_target_counter = 0
        self.update_target_rate = update_target_rate
        self.state_dim = state_dim

        # memory parameters
        self.replay_buffer: VanillaReplayBuffer = self.init_memory_buffer(memory_info)

        self.seed = seed
        self.tot_timestep = 0
        self.is_vanilla = is_vanilla



    # ====== Initialization functions ==================================



    def initialize_policy_estimator(self, params):
        return CustomNeuralNetwork(**params)

    def initialize_function_approximator(self, params):

        return CustomNeuralNetwork(**params)
        #self.critic = DQN(params)

    def init_memory_buffer(self, params: Dict) -> VanillaReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        return VanillaReplayBuffer(**params)

    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.critic_eval.set_seed(seed)
            self.critic_target.set_seed(seed)
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch([self.actor, self.critic_eval], type="grad")


    # ====== Memory functions ==========================================

    def update_target_net(self):
        # every n learning cycle, the target networks will be replaced 
        # with the eval networks
        if self.update_target_counter % self.update_target_rate == 0:
            self.critic_target.load_state_dict(
                self.critic_eval.state_dict())
        self.update_target_counter += 1

    def control(self):
        """ DO NOT USE!!! 

        :param state:
        :param reward:
        :return:
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()

        if self.replay_buffer.full:
            # getting batch data
            batch = self.replay_buffer.sample()
            if self.is_continuous:
                cur_oa = self._concat_obs_action(batch.observations, batch.actions)
                target_actions = self.actor(batch.next_observations).detach()
                next_oa = self._concat_obs_action(batch.next_observations, target_actions)
                q_next = self.critic_target(next_oa).detach()
                q_next = (1.0 - batch.dones.float()) * q_next
                y = batch.rewards + self.γ * q_next
            else:
                prev_state_value = self.critic_eval(batch.observations)
                state_value = self.critic_target(batch.next_observations)
                nu_state_value = torch.zeros(state_value.shape)
                nu_state_value = torch.masked_fill(state_value, batch.dones, 0.0)

                δ = batch.rewards + self.γ * nu_state_value.detach() - prev_state_value.detach()
                value_loss = - prev_state_value * δ 
                value_loss = value_loss.mean()
                self.critic_eval.backpropagate(value_loss)

                # plot the policy entropy
                probs = self.actor(batch.observations)
                #entropy = -(np.sum(probs * np.log(probs)))
                entropy = -(
                    torch.sum(
                        probs * torch.log(probs), dim=1, keepdim=True
                    ).mean()
                )
                
                # compute actor loss
                logprob = - torch.log(self.actor(
                    batch.observations).gather(1, batch.actions.long()))
                loss = logprob * δ 
                loss = loss.mean()
                self.actor.backpropagate(loss)

            self.logger.log({
                'Agent info/critic loss': value_loss,
                'Agent info/actor loss': loss,
                'Agent info/entropy': entropy},
                type= "agent")

    def vanilla_control(self, state, reward, is_terminal_state):
        
        obs_val = self.critic_eval(self.previous_state)
        if is_terminal_state:
            next_obs_val = torch.tensor([0])
        else:
            next_obs_val = self.critic_eval(state)
        advantage = reward + self.γ * next_obs_val.detach() - obs_val.detach()

        value_loss = - obs_val * advantage 
        self.critic_eval.backpropagate(value_loss)
            
        if self.is_continuous:
            # TODO make that work for multidimensional actions when necessary.
            logprob = - self.actor_cont(self.previous_state).log_prob(
                torch.Tensor([self.previous_action]))
            #prob = 1 / (math.sqrt(math.pi * 2) * std) * \
            #    torch.exp(-((mu - self.previous_action) ** 2) / (2 * std ** 2))
            #logprob = - torch.log(prob)
            #logprob = torch.normal(mu, std).log_prob(
            #    self.previous_action)
            #action = mu + torch.randn(mu.shape) * std
            #action = action.clamp(-1, 1)
            #action = action.detach()
            # TODO: compute the loss.
        else:
            # plot the policy entropy
            probs = self.actor(state).detach().numpy()
            entropy = -(np.sum(probs * np.log(probs)))
            logprob = - torch.log(self.actor(self.previous_state)[self.previous_action])
        loss = logprob * advantage 
        self.actor.backpropagate(loss)
        
        
        self.logger.log({
                'Agent info/critic loss': value_loss,
                'Agent info/actor loss': loss#,
                #'Agent info/entropy': entropy
                },
                type= "agent")



    # ====== Action choice related functions ===========================



    def choose_action(self, state): # TODO fix first if
        if self.is_continuous:  
            action = self.actor_cont(state).sample().clamp(-1, 1)
            #action_chosen = torch.normal(mu, std).clamp(-1, 1)
            return action.item()
        else:
            action_probs = Categorical(self.actor(state))
            action = action_probs.sample()
            return action.item()



    # ====== Agent core functions ======================================



    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(
            self.previous_state, 
            self.previous_action, 
            reward, 
            state, 
            False
        )
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        if self.is_vanilla:
            self.vanilla_control(state, reward, False)
        else:
            self.control()
        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition( 
            self.previous_state, 
            self.previous_action, 
            reward, 
            state, 
            True 
        )
        if self.is_vanilla:
            self.vanilla_control(state, reward, True)
        else:
            self.control()
        

    def get_state_value_eval(self, state):
        if self.num_actions > 1:
            state_value = self.actor(state).data
        else: 
            state_value = self.critic_eval(state).data
        return state_value
    
    def adjust_dims(self, state_dim:int, action_dim:int):
        """ Called when the agent dimension doesn't fit the environment 
        dimension. Reinitialize some parts of the agent so they fit the 
        environment.
        """
        self.state_dim = state_dim
        self.num_actions = action_dim
        if self.is_continuous:
            self.actor.reinit_layers(state_dim, action_dim * 2)
            self.critic_eval.reinit_layers(state_dim, 1)
            self.critic_target.reinit_layers(state_dim, 1)
        else:
            self.actor.reinit_layers(state_dim, action_dim)
            self.critic_eval.reinit_layers(state_dim, 1)
            self.critic_target.reinit_layers(state_dim, 1)
        self.replay_buffer.correct(state_dim, action_dim)
        self.logger.wandb_watch([self.actor, self.critic_eval])

    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        """ for plotting purposes only in continuous probe environment. 
        """
        #state = torch.cat((state, state)).unsqueeze(1)
        #state = (state.unsqueeze(1) * torch.ones(len(actions))).T
        #state_action = torch.cat((state, actions.unsqueeze(1)),1)
        comp_action = self.actor(state)
        mu = comp_action[0]
        std = comp_action[1]
        action_probs = actions.detach().apply_(lambda x: norm.pdf(x, mu.item(), std.item()))
        #action_prob = scp.stats.norm.pdf(actions, mu, std)

        #= self.critic_eval(state).data
        return action_probs

    def get_action_probs(self, state:torch.Tensor, actions:torch.Tensor):
        pass

    def _concat_obs_action(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action), 1)#.unsqueeze(1)),1)
        return obs_action
    
    def actor_cont(self, state):
        comp_action = self.actor(state)
        mu = comp_action[0]
        std = comp_action[1]
        action_probs = torch.distributions.Normal(mu, std)
        return action_probs