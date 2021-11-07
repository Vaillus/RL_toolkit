from utils import set_random_seed
import torch
from typing import Any, Dict, Optional, Type

from CustomNeuralNetwork import CustomNeuralNetwork
from modules.replay_buffer import PerfoReplayBuffer
from modules.logger import Logger

class PerfoAgent:
    def __init__(
        self, 
        nn_info: Dict[str, Any],
        memory_info: Dict[str, Any],
        seed: Optional[int] = 0,
        num_actions: Optional[int] = 1,
        state_dim: Optional[int] = 1, 
        device = None
    ):
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.nn: CustomNeuralNetwork = self.init_nn(nn_info, device=device)
        self.replay_buffer = self.init_memory_buffer(memory_info)

        self.seed = seed
        self.discount_factor = 0.9

        self.previous_state = None
        self.previous_action = None

    # ====== Initialization functions ==================================
     
    def init_nn(self, params: Dict, device) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(
            **params, 
            input_dim=self.state_dim, 
            output_dim=self.num_actions).to(device)
    
    def init_memory_buffer(self, params: Dict):
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        return PerfoReplayBuffer(**params)
    
    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.nn.set_seed(seed)
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch(self.nn, type="grad")
    

    def control(self):
        batch = self.replay_buffer.sample()
        #batch_actions = torch.unsqueeze((batch.actions == 1).nonzero()[:,1],1)
        q_eval = self.nn(batch.observations).gather(0, batch.actions.long())
        next_action_values = self.nn(batch.next_observations).detach()
        q_next = torch.unsqueeze(next_action_values.max(1)[0], 1)
        q_target = batch.rewards + self.discount_factor * q_next
        lossf = torch.nn.MSELoss()
        loss = lossf(q_target, q_eval)
        self.nn.backpropagate(loss)


    # ====== Action choice related functions ===========================

    def choose_action(self, obs) -> int:
        action_values = self.get_action_values(obs)
        action = torch.argmax(action_values)
        return action.item()
    
    def get_action_values(self, obs):
        # Compute action values from the eval net
        return self.nn(obs)

    # ====== Agent core functions ======================================

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.fill(self.previous_state, self.previous_action, reward, state, False)
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.control()
        #self.vanilla_control(state, reward, False)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        print("no end function for this agent")

    def adjust_dims(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.num_actions = action_dim
        self.nn.reinit_layers(state_dim, action_dim)
        self.replay_buffer.correct(state_dim, action_dim)
        self.logger.wandb_watch(self.nn, type="grad")

