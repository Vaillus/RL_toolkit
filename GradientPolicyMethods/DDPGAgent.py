from CustomNeuralNetwork import CustomNeuralNetwork
from utils import set_random_seed
import numpy as np
import torch
from memory_buffer import ReplayBuffer, ReplayBufferSamples

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:
    def __init__(self, params={}):
        # parameters to be set from params dict
        self.epsilon = None
        self.num_actions = None
        self.is_greedy = None
        self.function_approximator = None
        self.is_vanilla = None

        # parameters not set at initilization
        self.previous_action = None
        self.previous_state = None
        self.seed = None

        # neural network parameters
        self.eval_net = None # to be deleted
        self.target_net = None # to be deleted
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None

        self.update_target_rate = None
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        # learning parameters
        self.discount_factor = None
        # memory parameters
        self.replay_buffer = None
        # TODO : I'll delete those
        self.memory_size = None
        self.memory = None
        self.memory_counter = 0
        self.batch_size = None

        self.min_action = None # TODO : will probably get rid of these two.
        self.max_action = None
        self.target_policy_noise = None
        self.target_noise_clip = None

        self.writer = None
        self.tot_timestep = 0

        self.set_params_from_dict(params)

        self.set_other_params()

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.epsilon = params.get("epsilon", 0.9)
        self.num_actions = params.get("num_actions", 0)
        self.is_greedy = params.get("is_greedy", False)
        self.is_vanilla = params.get("is_vanilla", False)
        self.init_seed(params.get("seed", None))

        self.state_dim = params.get("state_dim", 4)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.discount_factor = params.get("discount_factor", 0.995)
        self.init_seed(params.get("seed", None))
        self.init_actor(params.get("policy_estimator_info"))
        self.init_critic(params.get("function_approximator_info"))

        replay_buffer_params = params.get("memory_info", {})
        self.init_memory_buffer(replay_buffer_params)

        self.memory_size = params.get("memory_size", 200)
        self.batch_size = params.get("batch_size", 64)

        self.min_action = params.get("min_action", 0.0) # will probably get rid of these two.
        self.max_action = params.get("max_action", 0.0)
        self.target_policy_noise = params.get("target_policy_noise", 0.2)
        self.target_noise_clip = params.get("target_noise_clip", 0.5)

    def set_other_params(self):
        # two slots for the states, + 1 for the reward an the last for 
        # the action (per memory slot)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 3))
    
    def init_actor(self, nn_params):
        self.actor = CustomNeuralNetwork(nn_params)
        self.actor_target = CustomNeuralNetwork(nn_params)
    
    def init_critic(self, params):
        self.critic = CustomNeuralNetwork(params)
        self.critic_target = CustomNeuralNetwork(params)
    
    def init_memory_buffer(self, params):
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        self.replay_buffer = ReplayBuffer(params)

    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.function_approximator.set_seed(seed)

    # ====== Action choice related functions ===========================

    def choose_action(self, action_values):
        """ choosing the action according to the strategy of the agent.
        Since the action is continuous and single, there is no set of actions
        to choose from. The function therefore just returns the action value.
        """
        if self.is_greedy:
            pass # where does it choose action?
            #action_chosen = np.argmax(action_values)
        else:
            pass # may find an alternative to egreedy
        # returning the chosen action and its value
        return action_values

    # ====== Control related functions =================================

    def control(self):
        self.update_weights()
    

    # ====== Agent core functions ======================================

    def start(self, state):
        # getting actions
        action_values = self.get_action_value(state)
        # choosing the action to take
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_state = state

        return current_action

    def step(self, state, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(self.previous_state, self.previous_action, 
                                            reward, state, False)
        # getting the action values from the function approximator
        action_values = self.get_action_value(state)
        # choosing an action
        numpy_action_values = action_values.clone().detach().numpy() # TODO : check if still relevant
        current_action = self.choose_action(numpy_action_values)

        self.control()

        self.previous_action = current_action
        self.previous_state = state
        
        return current_action

    def end(self, state, reward):
        self.replay_buffer.store_transition(self.previous_state, 
                                    self.previous_action, reward, state, True)
        self.control()

    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        action_value = self.eval_net(state)
        noise = 0 # normal distrib, for exploration
        action_value = self.eval_net(state) + noise
        action_value = torch.clamp(action_value, self.min_action, self.max_action)
        return action_value

    # === parameters update functions ==================================

    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        if self.update_target_counter % self.update_target_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_target_counter += 1

    def compute_loss(self, batch:ReplayBufferSamples):
        # value of the action being taken at the current timestep
        q_eval = self.critic(batch.observations).gather(1, batch.actions.long())
        # values of the actions at the next step
        q_next = self.critic_target(batch.next_observations).detach()
        q_next = self._zero_terminal_states(q_next, batch.next_observations)
        noise = self._create_noise_tensor(batch.actions)
        q_next += noise
        # Q containing only the max value of q in next step
        q_target = batch.rewards + self.discount_factor * q_next
        # computing the loss
        loss = self.loss_func(q_eval, q_target)

        
        # residual variance for plotting purposes (not sure if it is correct)
        q_res = self.target_net(batch.observations).gather(1, batch.actions.long())
        res_var = torch.var(q_res - q_eval) / torch.var(q_res)
        self.writer.add_scalar("Agent info/residual variance", res_var, self.tot_timestep)
        
        return loss

    def update_weights(self):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()
        # we can start learning when the memory is full
        if (self.memory_counter > self.memory_size):
            # getting batch data
            sample = self.replay_buffer.sample()
            # Compute and backpropagate loss
            loss = self.compute_loss(sample)
            self.writer.add_scalar("Agent info/loss", loss, self.tot_timestep)
            self.eval_net.backpropagate(loss)
    
    def get_state_value_eval(self, state):
        """for plotting purposes only?
        """
        first_action_value = self.critic(state, 0.25).data
        sec_action_value = self.critic(state, 0.75).data
        return [first_action_value, sec_action_value]
    
    def _zero_terminal_states(self,  q_values: torch.Tensor,
                                     dones:torch.Tensor) -> torch.Tensor:
        """ Zeroes the q values at terminal states
        """
        nu_q_values = torch.zeros(q_values.shape)
        nu_q_values = torch.masked_fill(q_values, dones, 0.0)
        return nu_q_values
    
    def _create_noise_tensor(self, tensor):
        # create the nois tensor filled with normal distribution
        noise = tensor.clone().data.normal_(0, self.target_policy_noise)
        # clip the normal distribution
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        return noise
