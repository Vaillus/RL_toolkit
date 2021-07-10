from CustomNeuralNetwork import CustomNeuralNetwork
from utils import set_random_seed
import numpy as np
import torch
from memory_buffer import ReplayBuffer, ReplayBufferSamples
import wandb

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:
    def __init__(self, params={}):
        self.num_actions = None
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
        self.discount_factor = None
        # memory parameters
        self.replay_buffer = None

        self.min_action = None # TODO : will probably get rid of these two.
        self.max_action = None
        self.target_policy_noise = None
        self.target_noise_clip = None

        self.tot_timestep = 0

        self.set_params_from_dict(params)

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.init_seed(params.get("seed", None))
        self.num_actions = params.get("num_actions", 1)
        self.state_dim = params.get("state_dim", 4)
        self.update_target_rate = params.get("update_target_rate", 50)
        self.discount_factor = params.get("discount_factor", 0.995)
        self.init_seed(params.get("seed", None))
        self.init_actor(params.get("policy_estimator_info"))
        self.init_critic(params.get("function_approximator_info"))

        replay_buffer_params = params.get("memory_info", {})
        self.init_memory_buffer(replay_buffer_params)

        self.min_action = params.get("min_action", 0.0) # will probably get rid of these two.
        self.max_action = params.get("max_action", 0.0)
        self.target_policy_noise = params.get("target_policy_noise", 0.2)
        self.target_noise_clip = params.get("target_noise_clip", 0.5)
    
    def init_actor(self, nn_params):
        self.actor = CustomNeuralNetwork(nn_params)
        self.actor_target = CustomNeuralNetwork(nn_params)
    
    def init_critic(self, params):
        self.critic = CustomNeuralNetwork(params)
        self.critic_target = CustomNeuralNetwork(params)
    
    def init_memory_buffer(self, params):
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        self.replay_buffer = ReplayBuffer(**params)

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
        if self.replay_buffer.full:
            # getting batch data
            batch = self.replay_buffer.sample()

            # value of the action being taken at the current timestep
            batch_oa = self._concat_obs_action(batch.observations, batch.actions)
            q_eval = self.critic(batch_oa)
            # values of the actions at the next step
            q_next = self.critic_target(batch_oa)
            q_next = self._zero_terminal_states(q_next, batch.next_observations)
            noise = self._create_noise_tensor(batch.actions)
            q_next += noise
            # Q containing only the max value of q in next step
            q_target = batch.rewards + self.discount_factor * q_next
            # computing the loss
            critic_loss = self.loss_func(q_eval, q_target.detach())

            self.critic.backpropagate(critic_loss)
            
            actions = self.actor(batch.observations)
            batch_oa = self._concat_obs_action(batch.observations, actions)
            actor_loss = - self.critic(batch_oa).mean()
            self.actor.backpropagate(actor_loss)

            # residual variance for plotting purposes (not sure if it is correct)
            #q_res = self.target_net(batch.observations).gather(1, batch.actions.long())
            #res_var = torch.var(q_res - q_eval) / torch.var(q_res)

            wandb.log({
                "Agent info/critic loss": critic_loss,
                "Agent info/actor loss": actor_loss
            })


    def _concat_obs_action(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action), 1)#.unsqueeze(1)),1)
        return obs_action
    
<<<<<<< HEAD
    def get_state_value_eval(self, state:torch.Tensor):
        """ for plotting purposes only?
        """
        first_action = torch.tensor([-0.5])
        sec_action = torch.tensor([0.5])
        first_stt_act = torch.cat((state, first_action))
        sec_stt_act = torch.cat((state, sec_action))
        first_action_value = self.critic(first_stt_act).detach().data
        sec_action_value = self.critic(sec_stt_act).detach().data
        return np.mean([first_action_value, sec_action_value])
=======
    def get_action_value_eval(self, state:torch.Tensor):
        """for plotting purposes only?
        """
        action = np.random.uniform(-1, 1, 1)
        action = torch.Tensor(action)
        state_action = torch.cat((state, action))
        action_value = self.critic(state_action).detach().data
        return action_value
>>>>>>> f1a4c25872771ee7283256d321e37fe387fb3d35
    
    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        """ for plotting purposes only?
        """
        state = torch.cat((state, state)).unsqueeze(1)
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