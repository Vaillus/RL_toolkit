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

        # parameters not set at initilization
        self.previous_action = None
        self.previous_obs = None
        self.seed = None

        # neural network parameters
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None

        self.update_target_rate = None
        self.update_target_counter = 0
        self.loss_func = torch.nn.MSELoss()
        # learning parameters
        self.γ = None
        # memory parameters
        self.replay_buffer = None

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
    

    # ====== Agent core functions ======================================



    def start(self, obs):
        # getting actions
        current_action = self.get_action_value(obs)

        # saving the action and the tiles activated
        self.previous_action = current_action
        self.previous_obs = obs

        return current_action

    def step(self, obs, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(self.previous_obs, self.previous_action, 
                                            reward, obs, False)
        self.control()

        # getting the action value
        current_action = self.actor(obs) # I may need to convert that to numpy.
        # choosing an action
        self.previous_action = current_action
        self.previous_obs = obs
        
        return current_action

    def end(self, state, reward):
        self.replay_buffer.store_transition(self.previous_obs, 
                                    self.previous_action, reward, state, True)
        self.control()

    # === functional functions =========================================

    def get_action_value(self, obs, action):
        # Compute action values from the eval net
        noise = 0 # necessary, here. Don't think so.
        action_value = self.critic(obs, action) + noise
        #action_value = torch.clamp(action_value)
        return action_value



    # ====== Control related functions =================================

    def control(self):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_net()
        # we can start learning when the memory is full
        if self.replay_buffer.full:
            # get a batch of experience
            batch = self.replay_buffer.sample()
            noise = self._create_noise_tensor(batch.actions)
            next_actions = self.actor_target(batch.next_observations).detach()
            next_actions = (next_actions + noise).clamp(-1, 1)

            q_next = self.critic_target(batch.next_observations, next_actions)
            q_next = self._zero_terminal_states(q_next, batch.dones)
            q_target = batch.rewards + self.γ * q_next
            q_current = self.critic(batch.observations, batch.actions)
            # computing the critic loss
            critic_loss = self.loss_func(q_current, q_target)
            self.critic.backpropagate(critic_loss)
            self.writer.add_scalar("Agent info/critic loss", critic_loss, self.tot_timestep)

            # Actor loss:
            actions = self.actor(batch.observation)
            actor_loss = - self.critic(batch.observations, actions)
            self.actor.backpropagate(actor_loss) # are the gradients of the critic
            # used here? I hope not. If there is problem, put a function with 
            # torch.no_grad() in it.
            self.writer.add_scalar("Agent info/actor loss", actor_loss, self.tot_timestep)

            # residual variance for plotting purposes (not sure if it is correct)
            q_res = self.target_net(batch.observations).gather(1, batch.actions.long())
            res_var = torch.var(q_res - q_eval) / torch.var(q_res)
            self.writer.add_scalar("Agent info/residual variance", res_var, self.tot_timestep)

    
    def update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        if self.update_target_counter % self.update_target_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_target_counter += 1

    
    def get_state_value_eval(self, state):
        state_value = self.critic(state).data
        return state_value
    
    def _zero_terminal_states(  self, q_values: torch.FloatTensor,
                                dones:torch.BoolTensor) -> torch.FloatTensor:
        """ Zeroes the q values at terminal states
        """
        nu_q_values = torch.zeros(q_values.shape)
        nu_q_values = torch.masked_fill(q_values, dones, 0.0)
        return nu_q_values
    
    def _create_noise_tensor(self, batch_actions:torch.FloatTensor) -> torch.FloatTensor:
        # create the nois tensor filled with normal distribution
        noise = batch_actions.clone().data.normal_(0, self.target_policy_noise)
        # clip the normal distribution
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        return noise
