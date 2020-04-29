from DQN.DQN import *

class DQNAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None
        self.dqn = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None

        self.set_params_from_dict(params)

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.epsilon = params.get("epsilon", 0.9)
        self.num_actions = params.get("num_actions", 0)
        self.is_greedy = params.get("is_greedy", False)
        params["function_approximator"]["num_actions"] = self.num_actions
        self.initialize_dqn(params.get("function_approximator"))

    def initialize_dqn(self, params):
        self.dqn = DQN(params)

    # ====== Action choice related functions =======================================================

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
        # returning the chosen action and its value
        return action_chosen

    # ====== Control related functions =======================================================

    def control(self):
        self.dqn.compute_weights()

    # ====== Agent core functions =======================================================

    def start(self, state):
        # getting actions
        action_values = self.dqn.get_action_value(state)
        # choosing the action to take
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state

        return current_action


    def step(self, state, reward):
        # getting the action values from the function approximator
        action_values = self.dqn.get_action_value(state)

        # storing the transition in the function approximator memory for further use
        self.dqn.store_transition(self.previous_state, self.previous_action, reward, state)
        # choosing an action
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        self.control()

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        self.dqn.store_transition(self.previous_state, self.previous_action, reward, state)
        self.control()
