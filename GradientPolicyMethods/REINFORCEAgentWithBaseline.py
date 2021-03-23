from CustomNeuralNetwork import CustomNeuralNetwork as CustomNN
import numpy as np
import torch


class REINFORCEAgentWithBaseline:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.γ = None

        self.policy_estimator = None
        self.function_approximator = None
        self.is_continuous = None

        self.states = []
        self.actions = []
        self.rewards = []

        self.writer = None
        self.tot_timestep = 0

        self.set_params_from_dict(params)

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.γ = params.get("discount_factor", 0.9)
        self.is_continuous = params.get("is_continuous", False)
        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_baseline_network(params.get("function_approximator_info"))

    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNN(params)

    def initialize_baseline_network(self, params):
        self.function_approximator = CustomNN(params)

    # ====== Control related functions =================================

    def control(self):
        self.function_approximator.compute_weights()

    # ====== Action choice related functions ===========================

    def choose_action(self, state):
        if self.is_continuous:
            action_chosen = self.policy_estimator(state).detach().numpy()
        else:
            action_probs = self.policy_estimator(state).detach().numpy()
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
        # getting the action
        current_action = self.choose_action(state)
        self.save_transition(reward, state, current_action)
        return current_action

    def end(self, state, reward):
        self.save_transition(reward)

    def save_transition(self, reward, state=None, action=None):
        self.rewards.append(reward)
        if state is not None and action is not None:
            self.states = np.vstack((self.states, state))
            self.actions = np.append(self.actions, action)
        

    def learn_from_experience(self):
        # TODO: probleme: comme j'ai pas ajouté le dernier état à la listes des états, on ne prend pas en compte la
        # dernière transition dans la partie DQN.
        discounted_reward = 0
        reversed_episode = zip(self.rewards[::-1], self.states[::-1], self.actions[::-1])
        for reward, state, action in reversed_episode:
            state_value = self.function_approximator(state)
            discounted_reward = reward + self.γ * discounted_reward
            δ = self.γ * (discounted_reward - state_value.detach())
            
            value_loss = - state_value * δ
            #value_loss = discounted_reward - state_value
            self.function_approximator.optimizer.zero_grad()
            value_loss.backward()
            self.function_approximator.optimizer.step()
            self.writer.add_scalar("Agent info/critic loss", value_loss, self.tot_timestep)
            
            # plot the policy entropy
            probs = self.policy_estimator(state).detach().numpy()
            entropy = -(np.sum(probs * np.log(probs)))
            self.writer.add_scalar("Agent info/policy entropy", entropy, self.tot_timestep)
            
            # on prend le contraire de l'expression pour que notre loss 
            # pénalise au bon moment.
            loss = - torch.log(self.policy_estimator(state)[action]) * δ
            self.policy_estimator.optimizer.zero_grad()
            loss.backward()
            self.policy_estimator.optimizer.step()
            self.writer.add_scalar("Agent info/actor loss", loss, self.tot_timestep)
            





