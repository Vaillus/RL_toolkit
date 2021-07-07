from CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import torch
from torch.distributions import Categorical
import wandb

class ActorCriticAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.γ = None
        self.num_actions = None

        self.policy_estimator = None

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

        self.seed = None

        self.tot_timestep = 0

        self.set_params_from_dict(params)
        self.set_other_params()

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.γ = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        self.is_continuous = params.get("is_continuous", False)

        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_function_approximator(params.get(
            "function_approximator_info"))

        self.memory_size = params.get("memory_size", 200)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.state_dim = params.get("state_dim", 4)
        self.batch_size = params.get("batch_size", 32)

        self.seed = params.get("seed", None)

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 3))
        

    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    def initialize_function_approximator(self, params):
        #self.function_approximator = DQN(params)
        self.function_approximator_eval = CustomNeuralNetwork(params)
        self.function_approximator_target = CustomNeuralNetwork(params)

    # ====== Memory functions ==========================================

    def store_transition(self, state, action, reward, next_state, is_terminal):
        # store a transition (SARS') in the memory
        is_terminal = [is_terminal]
        transition = np.hstack((state, [action, reward], next_state, is_terminal))
        self.memory[self.memory_counter % self.memory_size, :] = transition
        self.incr_mem_cnt()
        
    def incr_mem_cnt(self):
        # increment the memory counter and resets it to 0 when reached 
        # the memory size value to avoid a too large value
        self.memory_counter += 1
        #if self.memory_counter == self.memory_size:
        #    self.memory_counter = 0

    def sample_memory(self):
        # Sampling some indices from memory
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        # Getting the batch of samples corresponding to those indices 
        # and dividing it into state, action, reward and next state
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.tensor(batch_memory[:, :self.state_dim]).float()
        batch_action = torch.tensor(batch_memory[:, 
            self.state_dim:self.state_dim + 1].astype(int)).float()
        batch_reward = torch.tensor(batch_memory[:, 
            self.state_dim + 1:self.state_dim + 2]).float()
        batch_next_state = torch.tensor(batch_memory[:, -self.state_dim-1:-1]).float()
        batch_is_terminal = torch.tensor(batch_memory[:, -1:]).bool()

        return batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal

    def update_target_net(self):
        # every n learning cycle, the target networks will be replaced 
        # with the eval networks
        if self.update_target_counter % self.update_target_rate == 0:
            self.function_approximator_target.load_state_dict(
                self.function_approximator_eval.state_dict())
        self.update_target_counter += 1

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
            batch_state, batch_action, batch_reward, batch_next_state, batch_is_terminal = self.sample_memory()
            
            prev_state_value = self.function_approximator_eval(batch_state)
            state_value = self.function_approximator_target(batch_next_state)
            nu_state_value = torch.zeros(state_value.shape)
            nu_state_value = torch.masked_fill(state_value, batch_is_terminal, 0.0)

            δ = batch_reward + self.γ * nu_state_value.detach() - prev_state_value.detach()

            value_loss = - prev_state_value * δ 
            value_loss = value_loss.mean()
            self.function_approximator_eval.optimizer.zero_grad()
            value_loss.backward()
            self.function_approximator_eval.optimizer.step()

            # plot the policy entropy
            probs = self.policy_estimator(state).detach().numpy()
            entropy = -(np.sum(probs * np.log(probs)))
            
            logprob = - torch.log(self.policy_estimator(
                batch_state).gather(1, batch_action.long()))
            loss = logprob * δ 
            loss = loss.mean()
            self.policy_estimator.optimizer.zero_grad()
            loss.backward()
            self.policy_estimator.optimizer.step()

            wandb.log({
                "Agent info/critic loss": value_loss,
                "Agent info/policy entropy": entropy,
                "Agent info/actor loss": loss
            })

    def vanilla_control(self, state, reward, is_terminal_state):
        prev_state_value = self.function_approximator_eval(self.previous_state)
        if is_terminal_state:
            cur_state_value = torch.tensor([0])
        else:
            cur_state_value = self.function_approximator_eval(state)
        δ = reward + self.γ * cur_state_value.detach() - prev_state_value.detach()

        value_loss = - prev_state_value * δ 
        self.function_approximator_eval.optimizer.zero_grad()
        value_loss.backward()
        self.function_approximator_eval.optimizer.step()
        

        # plot the policy entropy
        probs = self.policy_estimator(state).detach().numpy()
        entropy = -(np.sum(probs * np.log(probs)))
        

        logprob = - torch.log(self.policy_estimator(self.previous_state)[self.previous_action])
        loss = logprob * δ 
        self.policy_estimator.optimizer.zero_grad()
        loss.backward()
        self.policy_estimator.optimizer.step()
        
        wandb.log({
                "Agent info/critic loss": value_loss,
                "Agent info/policy entropy": entropy,
                "Agent info/actor loss": loss
            })


    # ====== Action choice related functions ===========================

    def choose_action(self, state): # TODO fix first if
        if self.is_continuous:  
            action_chosen = self.policy_estimator(state).detach().numpy()
            return action_chosen
        else:
            action_probs = Categorical(self.policy_estimator(state))
            action_chosen = action_probs.sample()
            return action_chosen.item()

    # ====== Agent core functions ======================================

    def start(self, state):
        # choosing the action to take
        current_action = self.choose_action(state)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):

        # storing the transition in the function approximator memory for further use
        self.store_transition(self.previous_state, self.previous_action, reward, state, False)

        # getting the action values from the function approximator
        current_action = self.choose_action(state)

        #self.control(state, reward)
        self.vanilla_control(state, reward, False)

        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def end(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.store_transition(self.previous_state, self.previous_action, reward, state, True)
        #self.control(state, reward)
        self.vanilla_control(state, reward, True)

    def get_state_value_eval(self, state):
        if self.num_actions > 1:
            state_value = self.policy_estimator(state).data
        else: 
            state_value = self.function_approximator_eval(state).data
        return state_value