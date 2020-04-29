from FunctionApproximator import *


class Agent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.learning_rate = None # TODO : passer au function approximator?
        self.discount_factor = None
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None

        self.control_method = None
        self.function_approximation_method = None
        self.function_approximator = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None
        self.previous_action_values = None # TODO : useless for now
        self.set_params_from_dict(params)

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.epsilon = params.get("epsilon", 0.0)
        self.num_actions = params.get("num_actions", 0)
        self.learning_rate = params.get("learning_rate", 0.9)
        self.is_greedy = params.get("is_greedy", False)
        self.control_method = params.get("control_method", 'sarsa')

        self.function_approximation_method = params.get("function_approximation_method", "tile coding")
        params["function_approximator"]["num_actions"] = self.num_actions
        self.initialize_function_approximator(params.get("function_approximator"))


    #def set_other_params(self):
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
            if self.function_approximation_method == "neural network":
                pass
                #max_action_value, _ = torch.max(action_values, -1)
                #chosen_action_value = action_values[current_action]
                #print(f'max action value : {max_action_value}')
                #previous_action_value = self.previous_action_values
                #print(f'previous action value : {previous_action_value}')
                                # self.function_approximator.get_action_value(self.previous_state, self.previous_action)
                #delta = (previous_action_value[self.previous_action] - (reward + self.discount_factor * chosen_action_value)) **2
                #print(f'delta : {delta}')
                delta = None
            else:
                previous_action_value = self.function_approximator.get_action_value(self.previous_state,
                                                                                    self.previous_action)
                if self.control_method == 'sarsa':
                    delta = reward + self.discount_factor * action_values[current_action] - previous_action_value
                elif self.control_method == 'q-learning':
                    if self.function_approximation_method == "neural network":
                        delta = reward + self.discount_factor * torch.max(action_values) - previous_action_value
                    else:
                        delta = reward + self.discount_factor * np.max(action_values) - previous_action_value
                elif self.control_method == 'expected sarsa':
                    if self.function_approximation_method == "neural network":
                        delta = reward + self.discount_factor * torch.mean(action_values) - previous_action_value
                    else:
                        delta = reward + self.discount_factor * np.mean(action_values) - previous_action_value
        #print(self.function_approximator.tile_coder)
        #print(delta)

        self.function_approximator.compute_weights(self.learning_rate, delta, self.previous_state, self.previous_action)


    # ====== Agent core functions =======================================================

    def agent_start(self, state):
        # getting actions
        action_values = self.function_approximator.get_action_value(state)
        # choosing the action to take
        if self.function_approximation_method == "neural network":
            numpy_action_values = action_values.clone().detach().numpy()
            current_action = self.choose_action(numpy_action_values)
        else:
            current_action = self.choose_action(action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state
        self.previous_action_values = action_values.clone().detach()


        return current_action


    def agent_step(self, state, reward):
        # getting the action values from the function approximator
        action_values = self.function_approximator.get_action_value(state)
        if self.function_approximation_method == "neural network":
            # storing the transition in the function approximator memory for further use
            self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)
            # choosing an action
            numpy_action_values = action_values.clone().detach().numpy()
            current_action = self.choose_action(numpy_action_values)
        else:
            current_action = self.choose_action(action_values)

        #print(f'action taken : {current_action}')
        self.control(reward, action_values, current_action)

        self.previous_action = current_action
        self.previous_state = state
        self.previous_action_values = action_values.clone().detach()

        return current_action

    def agent_end(self, state, reward):
        self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)















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
    agent.initialize_tile_coder(env=env)

    EPISODES = 100
    SHOW_EVERY = 10
    success_count = 0

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        action = agent.agent_start(state)
        acc_reward = 0
        while not done:
            new_state, reward, done, _ = env.step(action)
            acc_reward += reward
            #if episode % SHOW_EVERY == 0:
            #        env.render()

            if done:
                print(acc_reward)
                if new_state[0] >= env.goal_position:
                    agent.agent_end(reward)
                    success_count += 1
                else:
                    agent.agent_end(reward)
            else:
                action = agent.agent_step(new_state, reward)

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