from FunctionApproximator import *
from GradientPolicyMethods.PolicyEstimator import *

class REINFORCEAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.discount_factor = None
        self.num_actions = None

        self.policy_estimator = None

        self.states = []
        self.actions = []
        self.rewards = []

        self.set_params_from_dict(params)
        # self.set_other_params()

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 0)

        # self.function_approximation_method = params.get("function_approximation_method", "tile coding")
        params["policy_estimator_info"]["output_size"] = self.num_actions
        self.initialize_policy_estimator(params.get("policy_estimator_info"))

    def initialize_policy_estimator(self, params):
        self.policy_estimator = PolicyEstimator(params)

    # ====== Action choice related functions =======================================================

    def choose_action(self, action_probs):
        action_chosen = np.random.choice(len(action_probs), p=action_probs)
        return action_chosen

    def start(self, state):
        # getting actions
        action_values = self.policy_estimator.predict(state).detach().numpy()
        # choosing the action to take
        current_action = self.choose_action(action_values)
        self.states = np.array([state])
        self.actions = np.array([current_action])
        self.rewards = []

        return current_action

    def step(self, state, reward):
        # getting the action values from the function approximator
        action_values = self.policy_estimator.predict(state).detach().numpy()
        current_action = self.choose_action(action_values)
        self.rewards.append(reward)
        self.states = np.vstack((self.states, state))
        self.actions = np.append(self.actions, current_action)
        #self.episode_memory.extend((reward, state, current_action))

        return current_action

    def end(self, state, reward):
        self.rewards.append(reward)
        # self.states.extend(state)
        # self.episode_memory.extend((reward, state))



    def learn_from_experience(self):
        #self.policy_estimator.optimizer.zero_grad()
        discounted_reward = 0
        reversed_episode = zip(self.rewards[::-1], self.states[::-1], self.actions[::-1])
        for reward, state, action in reversed_episode:
            self.policy_estimator.optimizer.zero_grad()
            discounted_reward = reward + self.discount_factor * discounted_reward
            loss = - torch.log(self.policy_estimator.predict(state)[action]) * discounted_reward
            loss.backward()
            self.policy_estimator.optimizer.step()





