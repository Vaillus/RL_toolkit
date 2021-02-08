from CustomNeuralNetwork import CustomNeuralNetwork
import torch
import numpy as np
from torch.distributions import Categorical


class REINFORCEAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.γ = None
        self.policy_estimator = None
        self.is_continuous = None

        self.states = []
        self.actions = []
        self.rewards = []

        self.seed = None
        
        self.set_params_from_dict(params)

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.γ = params.get("discount_factor", 0.9)
        self.is_continuous = params.get("is_continuous", False)
        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.is_continuous = params.get("is_continuous", False)
        self.seed = params.get("seed", None)
        if self.seed:
            torch.manual_seed(self.seed)

    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    # ====== Action choice related functions ===========================

    def choose_action(self, state):
        if self.is_continuous:
            # TODO: I don't think that's correct
            action_probs = Categorical(self.policy_estimator(state))
        else:
            action_probs = Categorical(self.policy_estimator(state))
            action_chosen = action_probs.sample().numpy()
        return action_chosen

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)
        self.states = np.array([state])
        self.actions = np.array([current_action])
        self.rewards = []

        return current_action

    def step(self, state, reward):
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.rewards.append(reward)
        self.states = np.vstack((self.states, state))
        self.actions = np.append(self.actions, current_action)

        return current_action

    def end(self, state, reward):
        self.rewards.append(reward)

    def learn_from_experience(self):
        """ replays the episode backward and make gradient ascent over 
        the policy
        """
        #self.policy_estimator.optimizer.zero_grad()
        discounted_reward = 0
        reversed_episode = zip(self.rewards[::-1], self.states[::-1], self.actions[::-1])
        for reward, state, action in reversed_episode:
            self.policy_estimator.optimizer.zero_grad()
            discounted_reward = reward + self.γ * discounted_reward
            # on prend le contraire de l'expression pour que notre loss 
            # pénalise au bon moment.
            loss = - torch.log(self.policy_estimator(state)[action]) * discounted_reward
            loss.backward()
            self.policy_estimator.optimizer.step()





