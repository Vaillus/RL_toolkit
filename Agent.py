from FunctionApproximator import *


class Agent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.learning_rate = None # TODO : passer au function approximator?
        self.discount_factor = None
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None

        self.eligibility_traces = None
        self.trace_decay = None
        # self.weights = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        self.control_method = None
        self.function_approximation_method = None
        self.function_approximator = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None
        self.previous_action_values = None # TODO : useless for now
        self.set_params_from_dict(params)
        self.set_other_params()

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.epsilon = params.get("epsilon", 0.0)
        self.num_actions = params.get("num_actions", 0)
        self.learning_rate = params.get("learning_rate", 0.9)
        self.is_greedy = params.get("is_greedy", False)
        self.control_method = params.get("control_method", 'sarsa')

        self.function_approximation_method = params.get("function_approximation_method", "tile coding")
        params["function_approximator_info"]["num_actions"] = self.num_actions
        self.initialize_function_approximator(params.get("function_approximator_info"))

        self.trace_decay = params.get("trace_decay", 0.1)


    def set_other_params(self):
        self.eligibility_traces = np.zeros(self.function_approximator.weights.shape)
        #self.weights = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
        #self.learning_rate /= self.num_tilings

    def initialize_function_approximator(self, params):
        self.function_approximator = FunctionApproximator(params)

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

    def control(self, reward, action_values=None, current_action=None, last_state=False):
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
                delta = reward + self.discount_factor * np.mean(action_values) - previous_action_value

        # TODO : organize that later
        #self.function_approximator.compute_weights(self.learning_rate, delta, self.previous_state, self.previous_action)
        self.update_eligibility_traces()
        self.function_approximator.compute_weights_with_eligibility_traces(self.learning_rate, delta, self.eligibility_traces)

    def update_eligibility_traces(self):
        self.eligibility_traces = self.discount_factor * self.trace_decay * self.eligibility_traces
        tiles = self.function_approximator.tile_coder.get_activated_tiles(self.previous_state)
        self.eligibility_traces[self.previous_action, tiles] += 1

    # ====== Agent core functions =======================================================

    def start(self, state):
        # getting actions
        action_values = self.function_approximator.get_action_value(state)
        # choosing the action to take
        current_action = self.choose_action(action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state


        return current_action


    def step(self, state, reward):
        # getting the action values from the function approximator
        action_values = self.function_approximator.get_action_value(state)
        current_action = self.choose_action(action_values)

        #print(f'action taken : {current_action}')
        self.control(reward, action_values, current_action)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)
        self.control(reward, last_state=True)


if __name__ == "__main__":
    import gym

    env = gym.make("MountainCar-v0")

    agent = Agent({"num_actions": 3,
                   "is_greedy": False,
                   "epsilon": 0.95,
                   "learning_rate": 0.5,
                   "num_tiles": 4,
                   "num_tilings": 32,
                   "discount_factor": 1,
                   "control_method": "expected sarsa"})
    # agent.initialize_tile_coder(env=env)

    EPISODES = 100
    SHOW_EVERY = 10
    success_count = 0

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        action = agent.start(state)
        acc_reward = 0
        while not done:
            new_state, reward, done, _ = env.step(action)
            acc_reward += reward
            #if episode % SHOW_EVERY == 0:
            #        env.render()

            if done:
                print(acc_reward)
                if new_state[0] >= env.goal_position:
                    agent.end(reward)
                    success_count += 1
                else:
                    agent.end(reward)
            else:
                action = agent.step(new_state, reward)

        if episode % SHOW_EVERY == 0:
            print(" ")
            print(f'EPISODE: {episode}')
            #print(agent.weights.count(0.0))
            #print(f'    number of values in weights: {np.count_nonzero(agent.weights != 0)}')
            #print(f'    minimum value in weights: {agent.weights.min()}')
            #print(f'    maximum value in weights: {agent.weights.max()}')
            print(f'    pourcentage of success: {success_count/SHOW_EVERY * 100}%')
            success_count = 0
    env.close()