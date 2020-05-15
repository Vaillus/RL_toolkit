from FunctionApproximator import *


class Agent:
    def __init__(self, params):
        # parameters to be set from params dict
        self.learning_rate = None # TODO : passer au function approximator?
        self.discount_factor = None
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None

        self.traces_type = None
        self.traces = None
        self.trace_decay = None

        self.control_method = None
        self.function_approximation_method = None
        self.function_approximator = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None
        self.previous_action_values = None  # only for dutch traces
        self.set_params_from_dict(params)
        #self.set_other_params()

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.epsilon = params.get("epsilon", 0.0)
        self.num_actions = params.get("num_actions", 0)
        self.learning_rate = params.get("learning_rate", 0.9)
        self.is_greedy = params.get("is_greedy", False)
        self.control_method = params.get("control_method", 'sarsa')

        self.function_approximation_method = params.get("function_approximation_method", "tile coding")
        params["function_approximator_info"]["num_actions"] = self.num_actions
        self.initialize_function_approximator(params.get("function_approximator_info"))

        self.traces_type = params.get("traces_type", "no traces") # can also be "eligibility traces" or "dutch traces"
        self.trace_decay = params.get("trace_decay", 0.1)


    def set_other_params(self):
        self.traces = np.zeros(self.function_approximator.weights.shape)

    def initialize_function_approximator(self, params):
        self.function_approximator = FunctionApproximator(params)

    # ====== Action choice related functions =======================================================

    def _choose_epsilon_greedy_action(self, action_values):
        """
        choose the action with the maximum value with probability 1/epsilon, or a random action else.
        :param action_values: list containing the action values
        :return: int containing the value of the action chosen
        """
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
            action_chosen = self._choose_epsilon_greedy_action(action_values)
        # returning the chosen action and its value
        return action_chosen

    def _get_expected_sarsa_state_value(self, action_values):
        values = []
        max_action = np.argmax(action_values)
        for action_number, action_value in enumerate(action_values):
            if action_number == max_action:
                values.append(action_value * (self.epsilon + (1 - self.epsilon) / len(action_values)))
            else:
                values.append(action_value * (1 - self.epsilon) / len(action_values))
        return sum(values)

    # ====== Control related functions =======================================================

    def control(self, reward, action_values=None, current_action=None, last_state=False, current_state=None):
        delta = None
        if last_state is True:
            delta = reward
        else:
            previous_action_value = self.function_approximator.get_action_value(self.previous_state,
                                                                                self.previous_action)
            if self.control_method == 'sarsa':
                delta = reward + self.discount_factor * action_values[current_action] - previous_action_value
            elif self.control_method == 'q-learning':
                delta = reward + self.discount_factor * np.max(action_values) - previous_action_value
            elif self.control_method == 'expected sarsa':
                delta = reward + self.discount_factor * self._get_expected_sarsa_state_value(action_values) - \
                        previous_action_value

        # compute the weights of the function approximator:
        # without traces:
        if self.traces_type == "no traces":
            self.function_approximator.compute_weights(self.learning_rate, delta, self.previous_state,
                                                       self.previous_action)
        # with eligibility traces:
        elif self.traces_type == "eligibility traces":
            self.update_eligibility_traces()
            self.function_approximator.compute_weights_with_eligibility_traces(self.learning_rate, delta,
                                                                               self.traces)
        # with dutch traces:
        elif self.traces_type == "dutch traces":
            if last_state is False:
                self.update_dutch_traces()
                self.function_approximator.compute_weights_with_dutch_traces(self.learning_rate, delta,
                                                                             self.previous_state,
                                                                             self.previous_action,
                                                                             self.traces,
                                                                             action_values[current_action],
                                                                             self.previous_action_values[
                                                                                 self.previous_action])

    def update_eligibility_traces(self, cumulative=True):
        self.traces = self.discount_factor * self.trace_decay * self.traces
        tiles = self.function_approximator.tile_coder.get_activated_tiles(self.previous_state)
        if cumulative:
            self.traces[self.previous_action, tiles] += 1
        else:
            self.traces[self.previous_action, tiles] = 1

    def update_dutch_traces(self):
        tiles = self.function_approximator.tile_coder.get_activated_tiles(self.previous_state)
        tmp = 1 - np.sum(self.learning_rate * self.discount_factor * self.trace_decay * self.traces[
            self.previous_action, tiles])
        self.traces = self.discount_factor * self.trace_decay * self.traces
        self.traces[self.previous_action, tiles] += tmp


    # ====== Agent core functions =======================================================

    def start(self, state):
        # getting actions
        action_values = self.function_approximator.get_action_value(state)
        # choosing the action to take
        current_action = self.choose_action(action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state
        self.previous_action_values = action_values


        return current_action


    def step(self, state, reward):
        # getting the action values from the function approximator
        action_values = self.function_approximator.get_action_value(state)
        current_action = self.choose_action(action_values)
        #self.update_dutch_traces(state, current_action)
        #print(f'action taken : {current_action}')
        self.control(reward, action_values, current_action, current_state=state)

        self.previous_action = current_action
        self.previous_state = state
        self.previous_action_values = action_values

        return current_action


    def end(self, state, reward):
        # self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)
        self.control(reward, last_state=True, current_state=state)


