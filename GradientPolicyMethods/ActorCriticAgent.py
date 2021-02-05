from GradientPolicyMethods.PolicyEstimator import *
from DQN.DQN import *
from GradientPolicyMethods.BaselineNetwork import *
from DQN.CustomNeuralNetwork import *
import numpy as np
import torch

class ActorCriticAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.discount_factor = None
        self.num_actions = None

        self.policy_estimator_eval = None

        self.function_approximator_eval = None
        self.function_approximator_target = None

        self.previous_state = None
        self.previous_action = None
        #self.rewards = []
        self.is_continuous = None

        # memory parameters
        self.memory_size = None
        self.memory = []
        self.memory_counter = 0
        self.batch_size = None

        self.update_target_counter = 0
        self.update_target_rate = None
        self.state_dim = None

        self.set_params_from_dict(params)
        #self.set_other_params()

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        self.is_continuous = params.get("is_continuous", False)

        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_function_approximator(params.get(
            "function_approximator_info"))

        self.memory_size = params.get("memory_size", 200)
        self.update_target_rate = params.get("update_target_rate", 50)

    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    def initialize_function_approximator(self, params):
        #self.function_approximator = DQN(params)
        self.function_approximator_eval = CustomNeuralNetwork(params)
        self.function_approximator_target = CustomNeuralNetwork(params)

    # ====== Memory functions ==========================================

    def store_transition(self, state, action, reward, next_state):
        # store a transition (SARS') in the memory
        transition = np.hstack((state, [action, reward], next_state))
        self.memory[self.memory_counter, :] = transition
        self.incr_mem_cnt()
        
    def incr_mem_cnt(self):
        # increment the memory counter and resets it to 0 when reached 
        # the memory size value to avoid a too large value
        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.memory_counter = 0

    def sample_memory(self):
        # Sampling some indices from memory
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        # Getting the batch of samples corresponding to those indices 
        # and dividing it into state, action, reward and next state
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
        batch_action = torch.LongTensor(batch_memory[:, 
            self.state_dim:self.state_dim + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 
            self.state_dim + 1:self.state_dim + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_dim:])

        return batch_state, batch_action, batch_reward, batch_next_state

    def update_target_net(self):
        # every n learning cycle, the target networks will be replaced 
        # with the eval networks
        if self.update_target_counter % self.update_target_rate == 0:
            self.function_approximator_target.load_state_dict(
                self.function_approximator_eval.state_dict())
        self.update_target_counter += 1

    def control_complicated(self):
        self.function_approximator.update_target_net()
        if self.function_approximator.memory_counter > self.function_approximator.memory_size:
            # get sample batches of transitions
            batch_state, batch_action, batch_reward, batch_next_state = self.function_approximator.sample_memory()

            self.function_approximator.eval_net.optimizer.zero_grad()
            self.policy_estimator.optimizer.zero_grad()


            td_error = batch_reward + self.discount_factor * self.function_approximator.eval_net(batch_next_state) - \
                    self.function_approximator.eval_net(batch_state).detach()

            actor_loss = - torch.log(self.policy_estimator.predict(batch_state).gather(1, batch_action))

            loss = (actor_loss * td_error).sum()
            #loss = - torch.log(self.policy_estimator.predict(batch_state).gather(1, batch_action)) * delta

            loss.backward()
            self.policy_estimator.optimizer.step()
            self.function_approximator.eval_net.optimizer.step()

    def control(self, state, reward):
        """

        :param state:
        :param reward:
        :return:
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()

        if self.memory_counter > self.memory_size:
            # getting batch data
            batch_state, batch_action, batch_reward, batch_next_state = self.sample_memory()
            self.function_approximator_eval.optimizer.zero_grad()
            prev_state_value = self.function_approximator_eval.predict(batch_state)
            state_value = self.function_approximator_target.predict(batch_next_state)
            δ = batch_reward + self.discount_factor * state_value.detach() - prev_state_value.detach()
            value_loss = - prev_state_value * δ
            value_loss.backward()
            self.function_approximator_eval.optimizer.step()

            loss = - torch.log(self.policy_estimator.predict(self.previous_state)[self.previous_action]) * δ
            loss.backward()
            self.policy_estimator.optimizer.step()


        """
        self.function_approximator.optimizer.zero_grad()
        # computing the advantage.
        # The advantage is r + γ * St - St-1
        # 
        advantage = reward + self.discount_factor * self.function_approximator(torch.FloatTensor(state)) - \
                    self.function_approximator(torch.FloatTensor(self.previous_state))
        #current_state_value = self.function_approximator(torch.FloatTensor(self.previous_state))
        critic_loss = advantage.pow(2)
        # backpropagate the loss function
        critic_loss.backward()
        self.function_approximator.optimizer.step()


        self.policy_estimator.optimizer.zero_grad()
        # get the probabilities of previous actions
        probs = self.policy_estimator(self.previous_state)
        # get the action that was actually taken
        prev_action = torch.LongTensor([self.previous_action])
        # the probability of the action that was taken
        action_chosen_prob = torch.gather(probs, dim=0, index=prev_action)

        actor_loss = - torch.log(action_chosen_prob) * advantage.detach()
        actor_loss.backward()
        self.policy_estimator.optimizer.step()
        """
        """
        advantage = reward + self.discount_factor * self.function_approximator(torch.FloatTensor(state)) - \
            self.function_approximator(torch.FloatTensor(self.previous_state))
        critic_loss = advantage ** 2
        critic_loss.backward()
        self.function_approximator.optimizer.step()
        actor_loss = - torch.log(self.policy_estimator(self.previous_state))[self.previous_action] * \
                     advantage.detach()
        actor_loss.backward()
        #loss = actor_loss + critic_loss
        #loss.backward()
        self.policy_estimator.optimizer.step()
        """


    # ====== Action choice related functions ===========================

    def choose_action(self, state):
        if self.is_continuous:
            action_chosen = self.policy_estimator(state).detach().numpy()
        else:
            action_probs = self.policy_estimator(state).detach().numpy()
            action_chosen = np.random.choice(len(action_probs), p=action_probs)
        return action_chosen

    # ====== Agent core functions ======================================

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):

        # storing the transition in the function approximator memory for further use
        #self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)

        # getting the action values from the function approximator
        current_action = self.choose_action(state)

        self.control(state, reward)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # storing the transition in the function approximator memory for further use
        #self.function_approximator.store_transition(self.previous_state, self.previous_action, reward, state)
        self.control(state, reward)
