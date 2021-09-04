from utils import set_random_seed
import numpy as np
import torch
from copy import deepcopy
from typing import Any, Dict, Optional, Type

from CustomNeuralNetwork import CustomNeuralNetwork
from replay_buffer import ReplayBuffer
from logger import Logger

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent:
    def __init__(
        self, 
        memory_info: Dict[str, Any],
        function_approximator_info: Dict[str, Any],
        discount_factor: Optional[float] = 0.995,
        update_target_rate: Optional[int] = 50,
        is_vanilla: Optional[bool] = False,
        epsilon: Optional[float] = 0.9,
        state_dim: Optional[int] = 0,
        num_actions: Optional[int] = 0,
        seed: Optional[int] = 0,
        is_greedy: Optional[bool] = False
    ):
        # parameters to be set from params dict
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.is_greedy = is_greedy
        self.is_vanilla = is_vanilla

        # neural network parameters
        self.eval_net = None
        self.target_net = None
        self.update_target_rate = update_target_rate
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        # NN dimension parameters
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        self.logger = None

        self.seed = self.init_seed(seed)

        self.tot_timestep = 0
        self.previous_action = None
        self.previous_obs = None

        self.replay_buffer = self.init_memory_buffer(memory_info)
        self.init_nn(function_approximator_info)

    # ====== Initialization functions ==================================
    
    def init_nn(self, nn_params):
        self.target_net = CustomNeuralNetwork(**nn_params)
        self.eval_net = deepcopy(self.target_net)

    def init_seed(self, seed):
        if seed:
            set_random_seed(self.seed)
            return seed
        else:
            return None

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.target_net.set_seed(seed)
            self.eval_net.set_seed(seed)
            # TODO: set memory buffer seed?
    
    def init_memory_buffer(self, params) -> ReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        return ReplayBuffer(**params)
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch(self.eval_net)

    # ====== Action choice related functions ===========================

    def choose_epsilon_greedy_action(self, action_values):
        if np.random.uniform() < self.epsilon:
            action_chosen = np.argmax(action_values)
        else:
            action_chosen = np.random.randint(self.num_actions)
        return action_chosen

    def choose_action(self, action_values):
        # choosing the action according to the strategy of the agent
        if self.is_greedy:
            action_chosen = np.argmax(action_values)
        else:
            action_chosen = self.choose_epsilon_greedy_action(action_values)
        return action_chosen

    

    # ====== Agent core functions ======================================

    def start(self, obs):
        # getting actions
        action_values = self.get_action_value(obs)
        # choosing the action to take
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_obs = obs

        return current_action

    def step(self, obs, reward):
        # storing the transition in the function approximator memory for further use
        one_hot_action = torch.zeros(self.num_actions)
        one_hot_action[self.previous_action] = 1
        assert one_hot_action.sum() == 1.
        self.replay_buffer.store_transition(self.previous_obs, one_hot_action, 
                                            reward, obs, False)
        # getting the action values from the function approximator
        action_values = self.get_action_value(obs)

        # choosing an action
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)
        self.control()

        self.previous_action = current_action
        self.previous_obs = obs
        
        return current_action

    def end(self, obs, reward):
        one_hot_action = torch.zeros(self.num_actions)
        one_hot_action[self.previous_action] = 1
        self.replay_buffer.store_transition(self.previous_obs, one_hot_action, reward, obs, True)
        self.control()

    # === functional functions =========================================

    def get_action_value(self, obs):
        # Compute action values from the eval net
        return self.eval_net(obs)
   
    # === Control related functions ====================================

    def control(self):
        self._learn()
    
    def _learn(self):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """
        if self.replay_buffer.full:
            # every n learning cycle, the target network will be replaced 
            # with the eval network
            self.update_target_net()
            batch = self.replay_buffer.sample()
            # Compute and backpropagate loss
            loss = self.compute_loss(batch)
            self.eval_net.backpropagate(loss)
            self.logger.wandb_log({
                "Agent info/loss": loss
            })


    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_counter += 1
        if self.update_target_counter == self.update_target_rate:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.update_target_counter = 0

    def compute_loss(self, batch):
        batch_actions = torch.unsqueeze((batch.actions == 1).nonzero()[:,1],1)
        q_eval = self.eval_net(batch.observations).gather(1, batch_actions.long())
        #torch.unsqueeze((self.eval_net(batch.observations) * batch.actions).sum(1), 1)
        q_next = self.target_net(batch.next_observations).detach()
        q_next = (1.0 - batch.dones.float()) * q_next
        q_next = torch.unsqueeze(q_next.max(1)[0], 1)
        q_target = batch.rewards + self.discount_factor * q_next
        loss = self.loss_func(q_eval, q_target)
        return loss

    def get_state_value_eval(self, state):
        state_value = self.eval_net(state).data
        return state_value
    
    def adjust_dims(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.num_actions = action_dim
        self.target_net.reinit_layers(state_dim, action_dim)
        self.eval_net.reinit_layers(state_dim, action_dim)
        self.replay_buffer.correct(state_dim, action_dim)


if __name__ == "__main__":
    agent = DQNAgent()
