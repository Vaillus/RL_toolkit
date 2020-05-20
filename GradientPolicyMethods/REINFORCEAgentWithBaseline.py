from FunctionApproximator import *
from GradientPolicyMethods.PolicyEstimator import *
from DQN.DQN import *
from GradientPolicyMethods.BaselineNetwork import *


class REINFORCEAgentWithBaseline:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.discount_factor = None
        self.num_actions = None

        self.policy_estimator = None
        self.function_approximator = None

        self.states = []
        self.actions = []
        self.rewards = []
        self.is_continuous = None

        self.set_params_from_dict(params)
        # self.set_other_params()

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        self.is_continuous = params.get("is_continuous", False)

        params["policy_estimator_info"]["output_size"] = self.num_actions
        self.initialize_policy_estimator(params.get("policy_estimator_info"))

        #self.initialize_dqn(params.get("function_approximator_info"))
        self.initialize_baseline_network(params.get("function_approximator_info"))

    def initialize_policy_estimator(self, params):
        self.policy_estimator = PolicyEstimator(params)

    def initialize_dqn(self, params):
        self.function_approximator = DQN(params)

    def initialize_baseline_network(self, params):
        self.function_approximator = BaselineNetwork(params)

    # ====== Control related functions =======================================================

    def control(self):
        self.function_approximator.compute_weights()

    # ====== Action choice related functions =================================================

    def choose_action(self, state):
        if self.is_continuous:
            action_chosen = self.policy_estimator.predict(state).detach().numpy()
        else:
            action_probs = self.policy_estimator.predict(state).detach().numpy()
            action_chosen = np.random.choice(len(action_probs), p=action_probs)
        return action_chosen

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)
        self.states = np.array([state])
        self.actions = np.array([current_action])
        self.rewards = []

        return current_action

    def step(self, state, reward):
        # getting the action values
        current_action = self.choose_action(state)

        #self.function_approximator.store_transition(self.states[-1], self.actions[-1], reward, state)
        #self.control()

        self.rewards.append(reward)
        self.states = np.vstack((self.states, state))
        self.actions = np.append(self.actions, current_action)

        #self.episode_memory.extend((reward, state, current_action))

        return current_action

    def end(self, state, reward):
        #self.function_approximator.store_transition(self.states[-1], self.actions[-1], reward, state)
        #self.control()

        self.rewards.append(reward)
        # self.states.extend(state)
        # self.episode_memory.extend((reward, state))

    def learn_from_experience(self):
        # TODO: probleme: comme j'ai pas ajouté le dernier état à la listes des états, on ne prend pas en compte la
        # dernière transition dans la partie DQN.

        discounted_reward = 0
        last_state, last_action, last_reward = None, None, None

        reversed_episode = zip(self.rewards[::-1], self.states[::-1], self.actions[::-1])
        for reward, state, action in reversed_episode:
            #if last_state is not None:
            #    self.function_approximator.store_transition(last_state, last_action, discounted_reward, state)

            self.function_approximator.optimizer.zero_grad()
            state_value = self.function_approximator.predict(state)

            self.policy_estimator.optimizer.zero_grad()
            discounted_reward = reward + self.discount_factor * discounted_reward
            loss = - torch.log(self.policy_estimator.predict(state)[action]) * (discounted_reward - state_value)
            loss.backward()
            self.policy_estimator.optimizer.step()

            last_state = state
            last_action = action





