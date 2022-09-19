#from #GradientPolicyMethods.PolicyEstimator import *
from RL_toolkit.DQN.DQN import *
#from #GradientPolicyMethods.BaselineNetwork import *
from RL_toolkit.custom_nn import *


# TODO : I think this one is to be thrown away
class ActorCriticAgent:
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
        #self.set_other_params()

    # ====== Initialization functions =============================================================

    def set_params_from_dict(self, params={}):
        self.discount_factor = params.get("discount_factor", 0.9)
        self.num_actions = params.get("num_actions", 1)
        self.is_continuous = params.get("is_continuous", False)

        self.initialize_policy_estimator(params.get("policy_estimator_info"))
        self.initialize_function_approximator(params.get("function_approximator_info"))

    def initialize_policy_estimator(self, params):
        self.policy_estimator = CustomNeuralNetwork(params)

    def initialize_function_approximator(self, params):
        #self.function_approximator = DQN(params)
        self.function_approximator = CustomNeuralNetwork(params)

    # ====== Memory functions =============================================================

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
        self.function_approximator.optimizer.zero_grad()
        # computing the advantage
        advantage = reward + self.discount_factor * self.function_approximator(torch.FloatTensor(state)) - \
                    self.function_approximator(torch.FloatTensor(self.previous_state))
        #current_state_value = self.function_approximator(torch.FloatTensor(self.previous_state))
        critic_loss = advantage.pow(2)
        # backpropagate the loss function
        critic_loss.backward()
        self.function_approximator.optimizer.step()


        self.policy_estimator.optimizer.zero_grad()
        # get the probabilities of previous actions
        action = self.policy_estimator(self.previous_state)
        prev_action = torch.LongTensor([self.previous_action])
        #action_chosen_prob = torch.gather(probs, dim=0, index=prev_action)
        actor_loss = (action - prev_action) ** 2
        actor_loss.backward()
        self.policy_estimator.optimizer.step()

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


    # ====== Action choice related functions =======================================================

    def choose_action(self, state):
        if self.is_continuous:
            action_chosen = self.policy_estimator(state).detach().numpy()
        else:
            action_probs = self.policy_estimator(state).detach().numpy()
            action_chosen = np.random.choice(len(action_probs), p=action_probs)
        return action_chosen

    # ====== Agent core functions =======================================================

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
