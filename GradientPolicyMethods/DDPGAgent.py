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
        self.previous_obs = None
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

    def choose_action(self, obs:torch.Tensor):
        action = self.actor(obs)
        action = torch.clamp(action, -1, 1)
        return action


    # ====== Agent core functions ======================================

    def start(self, obs):
        current_action = self.actor(obs)
        self.previous_action = current_action
        self.previous_obs = obs

        return current_action

    def step(self, obs, reward):
        # storing the transition in the function approximator memory for further use
        self.replay_buffer.store_transition(self.previous_obs, self.previous_action, 
                                            reward, obs, False)
        # getting the action values from the function approximator
        current_action = self.actor(obs)
        self.control()
        self.previous_action = current_action
        self.previous_obs = obs
        
        return current_action

    def end(self, obs, reward):
        self.replay_buffer.store_transition(self.previous_obs, 
                                    self.previous_action, reward, obs, True)
        self.control()

    # === functional functions =========================================

    def get_action_value(self, state, action=None):
        # Compute action values from the eval net
        action_value = self.critic(state)
        noise = 0 # normal distrib, for exploration
        action_value = self.critic(state) + noise
        action_value = torch.clamp(action_value, self.min_action, self.max_action)
        return action_value

    # === parameters update functions ==================================

    def _update_target_net(self):
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self.update_target_counter += 1
        if self.update_target_counter == self.update_target_rate:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.update_target_counter = 0
        
    
    # ====== Control related functions =================================

    def control(self):
        self._learn()

    def _learn(self):
        """
        Updates target net, sample a batch of transitions and compute 
        loss from it
        :return: None
        """
        # every n learning cycle, the target network will be replaced 
        # with the eval network
        self._update_target_net()
        # we can start learning when the memory is full
        if (self.memory_counter > self.memory_size):
            # getting batch data
            batch = self.replay_buffer.sample()

            # value of the action being taken at the current timestep
            batch_oa = self._concat_obs_action(batch.observations, batch.actions)
            q_eval = self.critic(batch_oa)
            # values of the actions at the next step
            q_next = self.critic_target(batch.next_observations).detach()
            q_next = self._zero_terminal_states(q_next, batch.next_observations)
            noise = self._create_noise_tensor(batch.actions)
            q_next += noise
            # Q containing only the max value of q in next step
            q_target = batch.rewards + self.discount_factor * q_next
            # computing the loss
            critic_loss = self.loss_func(q_eval, q_target)

            self.critic.backpropagate(critic_loss)
            self.writer.add_scalar("Agent info/critic loss", critic_loss, self.tot_timestep)
            

            actions = self.actor(batch.observations)
            actor_loss = - self.critic(batch.observations, actions).mean()
            self.actor.backpropagate(actor_loss)
            self.writer.add_scalar("Agent info/actor loss", act, self.tot_timestep)

            # residual variance for plotting purposes (not sure if it is correct)
            q_res = self.target_net(batch.observations).gather(1, batch.actions.long())
            res_var = torch.var(q_res - q_eval) / torch.var(q_res)
            self.writer.add_scalar("Agent info/residual variance", res_var, self.tot_timestep)
        
            self.writer.add_scalar("Agent info/loss", loss, self.tot_timestep)
            self.critic.backpropagate(critic_loss)


    def _concat_obs_action(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action.unsqueeze(1)),1)
        return obs_action
    
    def get_state_value_eval(self, state:torch.Tensor):
        """for plotting purposes only?
        """
        first_action = torch.tensor([0.25])
        sec_action = torch.tensor([0.75])
        first_stt_act = torch.cat(state, first_action)
        sec_stt_act = torch.cat(state, sec_action)
        first_action_value = self.critic(first_stt_act).data
        sec_action_value = self.critic(sec_stt_act).data
        return [first_action_value, sec_action_value]
    
    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        """for plotting purposes only?
        """
        state = torch.cat((first_state, first_state)).unsqueeze(1)
        state_action = torch.cat((state, actions.unsqueeze(1)),1)
        action_values = self.critic(state_action).data
        return action_values
    
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

#%%
import torch
act = torch.ones(3)
state = torch.zeros((3,2))
print(act)
print(state)
torch.cat((state,act.unsqueeze(1)), 1)
# %%
import numpy as np
action = np.ones(3)
np.unsqueeze(action,1)
# %%
first_state = torch.tensor([-1])
actions = torch.tensor([0.25, 0.75])
a = torch.cat((first_state, first_state)).unsqueeze(1)
print(a)
print(actions.unsqueeze(1))
a = torch.cat((a, actions.unsqueeze(1)),1)
a
# %%
(first_state, actions.unsqueeze(1))
# %%
