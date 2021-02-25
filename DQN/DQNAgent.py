from DQN.DQN import *
from utils import set_random_seed
import numpy as np

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

        self.set_params_from_dict(params)

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.epsilon = params.get("epsilon", 0.9)
        self.num_actions = params.get("num_actions", 0)
        self.is_greedy = params.get("is_greedy", False)
        self.is_vanilla = params.get("is_vanilla", False)
        self.init_seed(params.get("seed", None))
        params["function_approximator_info"]["action_dim"] = self.num_actions
        self.initialize_dqn(params.get("function_approximator_info"))

    def initialize_dqn(self, params):
        self.function_approximator = DQN(params)

    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.function_approximator.set_seed(seed)

    # ====== Action choice related functions =======================================================

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

    # ====== Control related functions =======================================================

    def control(self):
        self.function_approximator.update_weights()

    # ====== Agent core functions ============================================================

    def start(self, state):
        # getting actions
        action_values = self.function_approximator.get_action_value(state)
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
            self.function_approximator.store_transition(self.previous_state,
                                                        self.previous_action, 
                                                        reward, 
                                                        state)

        # getting the action values from the function approximator
        action_values = self.function_approximator.get_action_value(state)
        # choosing an action
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        self.control()

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)
        self.control()
