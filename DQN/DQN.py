from DQN.CustomNeuralNetwork import *
import numpy as np
import torch

class DQN:
    def __init__(self, params={}):

        # neural network parameters
        self.eval_net = None
        self.target_net = None
        self.update_target_rate = None
        self.update_target_counter = 0
        self.loss_func = nn.MSELoss()
        # NN dimension parameters
        self.state_dim = None
        self.action_dim = None
        # learning parameters
        self.learning_rate = None
        self.discount_factor = None
        # memory parameters
        self.memory_size = None
        self.memory = None
        self.memory_counter = 0
        self.batch_size = None

        self.set_params_from_dict(params)

        self.set_other_params()

    def set_params_from_dict(self, params={}):
        self.state_dim = params.get("state_dim", 4)
        self.action_dim = params.get("action_dim", 2)
        self.memory_size = params.get("memory_size", 200)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.batch_size = params.get("batch_size", 128)
        self.discount_factor = params.get("discount_factor", 0.995)

        self.initialize_neural_networks(params.get("neural_nets_info"))

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 2))

    def initialize_neural_networks(self, nn_params):
        self.target_net, self.eval_net = (CustomNeuralNetwork(nn_params), 
        CustomNeuralNetwork(nn_params))

    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        if action is None:
            action_value = self.eval_net(state)
        else:
            action_value = self.eval_net(state)[action]
        return action_value

    # === memory related functions =====================================

    def store_transition(self, state, action, reward, next_state):
        # store a transition (SARS') in the memory
        transition = np.hstack((state, [action, reward], next_state))
        self.memory[self.memory_counter, :] = transition
        self.incr_mem_cnt()
        
    def incr_mem_cnt(self):
        # increment the memory counter and resets it to 0 when reached 
        # the memory size value to avoid a too large value
        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.memory_counter = 0

    def sample_memory(self):
        # Sampling some indices from memory
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        # Getting the batch of samples corresponding to those indices 
        # and dividing it into state, action, reward and next state
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
        batch_action = torch.LongTensor(batch_memory[:, 
            self.state_dim:self.state_dim + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 
            self.state_dim + 1:self.state_dim + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_dim:])

        return batch_state, batch_action, batch_reward, batch_next_state

    # === parameters update functions ==================================

    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        if self.update_target_counter % self.update_target_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_target_counter += 1

    def compute_loss(self, batch_state, batch_action, batch_reward, 
                        batch_next_state):
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
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.discount_factor * q_next.max(1)[0].view(
            self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        return loss

    def update_weights(self):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """""
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()
        # we can start learning when the memory is full
        if self.memory_counter > self.memory_size:
            # getting batch data
            batch_state, batch_action, batch_reward, batch_next_state = \
                self.sample_memory()

            # Compute and backpropagate loss
            loss = self.compute_loss(batch_state, batch_action, batch_reward, 
                                        batch_next_state)
            self.eval_net.backpropagate(loss)
