from RL_toolkit.custom_nn import CustomNeuralNetwork
import numpy as np
import torch
import torch.nn
from RL_toolkit.utils import set_random_seed

# obsolete

class vanilla_DQN:
    def __init__(self, params={}):

        # neural network parameters
        self.eval_net = None
        self.target_net = None
        self.update_target_rate = None
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        # NN dimension parameters
        self.state_dim = None
        self.action_dim = None
        # learning parameters
        self.learning_rate = None
        self.discount_factor = None

        self.seed = None

        self.set_params_from_dict(params)

    def set_params_from_dict(self, params={}):
        self.state_dim = params.get("state_dim", 4)
        self.action_dim = params.get("action_dim", 2)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.discount_factor = params.get("discount_factor", 0.995)
        self.init_seed(params.get("seed", None))
        self.initialize_neural_networks(params.get("function_approximator_info"))

    def initialize_neural_networks(self, nn_params):
        self.target_net, self.eval_net = (CustomNeuralNetwork(nn_params), 
        CustomNeuralNetwork(nn_params))
    
    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
    
    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(seed)
            self.target_net.set_seed(seed)
            self.eval_net.set_seed(seed)
            
    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        if action is None:
            action_value = self.eval_net(state)
        else:
            action_value = self.eval_net(state)[action]
        return action_value

# === parameters update functions ==================================

    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        if self.update_target_counter % self.update_target_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_target_counter += 1

    def compute_loss(self, state, action, reward, 
                        next_state):
        """
        Compute the loss
        :param batch_state: pytorch tensor of shape [batch_size, 
                                                            state_dim]
        :param batch_action: pytorch tensor of shape [batch_size, 1]
        :param batch_reward: pytorch tensor of shape [batch_size, 1]
        :param batch_next_state: pytorch tensor of shape [batch_size, 
                                                            state_dim]
        :return:
        """
        q_eval = self.eval_net(state)[action]
        q_next = self.target_net(next_state).detach()
        q_target = reward + self.discount_factor * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        return loss

    def update_weights(self, state, action, reward, next_state):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """""
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()
        # Compute and backpropagate loss
        loss = self.compute_loss( state, action, reward, next_state)
        self.eval_net.backpropagate(loss)
