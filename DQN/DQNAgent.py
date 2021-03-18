from CustomNeuralNetwork import CustomNeuralNetwork
from DQN.DQN import *
from DQN.vanilla_DQN import *
from utils import set_random_seed
import numpy as np
import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None
        self.function_approximator = None
        self.is_vanilla = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None
        self.seed = None

        # neural network parameters
        self.eval_net = None
        self.target_net = None
        self.update_target_rate = None
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        # NN dimension parameters
        self.state_dim = None
        # learning parameters
        self.learning_rate = None
        self.discount_factor = None
        # memory parameters
        self.memory_size = None
        self.memory = None
        self.memory_counter = 0
        self.batch_size = None

        self.writer = None

        self.set_params_from_dict(params)

        self.set_other_params()

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.epsilon = params.get("epsilon", 0.9)
        self.num_actions = params.get("num_actions", 0)
        self.is_greedy = params.get("is_greedy", False)
        self.is_vanilla = params.get("is_vanilla", False)
        self.init_seed(params.get("seed", None))

        self.state_dim = params.get("state_dim", 4)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.discount_factor = params.get("discount_factor", 0.995)
        self.init_seed(params.get("seed", None))
        self.initialize_neural_networks(params.get("function_approximator_info"))

        self.memory_size = params.get("memory_size", 200)
        self.batch_size = params.get("batch_size", 64)

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 2))
    
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
            set_random_seed(self.seed)
            self.function_approximator.set_seed(seed)

    # ====== Action choice related functions ===========================

    def choose_epsilon_greedy_action(self, action_values):
        if np.random.uniform() < self.epsilon:
            action_chosen = np.argmax(action_values)
        else:
            action_chosen = np.random.randint(self.num_actions)
        #action_chosen = np.zeros(len(action_values))
        #action_chosen[action_chosen_id] = 1
        return action_chosen

    def choose_action(self, action_values):
        # choosing the action according to the strategy of the agent
        if self.is_greedy:
            action_chosen = np.argmax(action_values)
        else:
            action_chosen = self.choose_epsilon_greedy_action(action_values)
        # returning the chosen action and its value
        return action_chosen

    # ====== Control related functions =================================

    def control(self):
        self.update_weights()
    
    #def vanilla_control(self, state, action, reward, next_state):
    #    self.update_weights(state, action, reward, next_state)

    # ====== Agent core functions ======================================

    def start(self, state):
        # getting actions
        action_values = self.get_action_value(state)
        # choosing the action to take
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):
        # storing the transition in the function approximator memory for further use
        if not self.is_vanilla:
            self.store_transition(self.previous_state,
                                                        self.previous_action, 
                                                        reward, 
                                                        state, False)

        # getting the action values from the function approximator
        action_values = self.get_action_value(state)

        # choosing an action
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        if self.is_vanilla:
            self.vanilla_control()
        else:
            self.control()

        self.previous_action = current_action
        self.previous_state = state
        
        return current_action

    def end(self, state, reward):
        self.store_transition(self.previous_state, self.previous_action, reward, state, True)
        if self.is_vanilla:
            self.vanilla_control()
        else:
            self.control()

    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        if action is None:
            action_value = self.eval_net(state)
        else:
            action_value = self.eval_net(state)[action]
        return action_value

    # === memory related functions =====================================

    def store_transition(self, state, action, reward, next_state, is_terminal):
        # store a transition (SARS' + is_terminal) in the memory
        transition = np.hstack((state, [action, reward], next_state, is_terminal))
        self.memory[self.memory_counter % self.memory_size, :] = transition
        self.incr_mem_cnt()
        
    def incr_mem_cnt(self):
        # increment the memory counter and resets it to 0 when reached 
        # the memory size value to avoid a too large value
        self.memory_counter += 1
        #if self.memory_counter == self.memory_size:
        #    self.memory_counter = 0

    def sample_memory(self):
        # Sampling some indices from memory
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        # Getting the batch of samples corresponding to those indices 
        # and dividing it into state, action, reward and next state
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.tensor(batch_memory[:, :self.state_dim]).float()
        batch_action = torch.tensor(batch_memory[:, 
            self.state_dim:self.state_dim + 1].astype(int)).float()
        batch_reward = torch.tensor(batch_memory[:, 
            self.state_dim + 1:self.state_dim + 2]).float()
        batch_next_state = torch.tensor(batch_memory[:, -self.state_dim-1:-1]).float()
        batch_is_terminal = torch.tensor(batch_memory[:, -1]).bool()

        return batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal

    # === parameters update functions ==================================

    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        if self.update_target_counter % self.update_target_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_target_counter += 1

    def compute_loss(self, batch_state, batch_action, batch_reward, 
                        batch_next_state, batch_ter_state):
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
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action.long())
        #q_eval = self.eval_net(batch_state.item())[action.item()]
        q_next = self.target_net(batch_next_state).detach()
        nu_q_next = torch.zeros(q_next.shape)
        nu_q_next = torch.masked_scatter(q_next, batch_ter_state)
        q_target = batch_reward + self.discount_factor * nu_q_next.max(1)[0].view(
            self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        res_var = torch.var(q_target - q_eval) / torch.var(q_target)
        self.writer.add_scalar("residual variance", res_var)
        self.writer.add_scalar("action value", q_eval.mean())
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
        if (self.memory_counter > self.memory_size):
            # getting batch data
            batch_state, batch_action, batch_reward, batch_next_state, batch_ter_state = \
                self.sample_memory()

            # Compute and backpropagate loss
            loss = self.compute_loss(batch_state, batch_action, batch_reward, 
                                        batch_next_state, batch_ter_state)
            self.writer.add_scalar("loss", loss)
            self.eval_net.backpropagate(loss)

if __name__ == "__main__":
    agent = DQNAgent()
