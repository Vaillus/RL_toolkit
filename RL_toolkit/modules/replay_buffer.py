import numpy as np
import torch
from typing import NamedTuple, Optional, List, Union
from collections import namedtuple

from RL_toolkit.custom_nn import CustomNeuralNetwork

import cProfile
import pstats
from copy import deepcopy


class BaseReplayBuffer:
    """ This class is used as the base for the other classes. It has all
    the basic components of a replay buffer.
    """
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        size:Optional[int] = 200
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = size
        self.pos = 0
        self.full = False
        self.observations = [] #np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions = [] #np.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = [] #np.zeros((self.size,1), dtype=np.float32)
        self.next_observations = []#np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.dones = [] #np.zeros((self.size,1), dtype=np.bool)
    
    def _incr_mem_cnt(self) -> None:
        """ Increment the memory counter and resets it to 0 when reached 
        the memory size value to avoid a too large value
        """
        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0



# === Vanilla replay buffer (as used in vanilla DQN) ===================



class VanillaReplayBufferSamples(NamedTuple): # Not sure how I would use it.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

# TODO: try to replace namedtuple with ReplayBufferSamples
VanillaReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards'])

class VanillaReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        size:Optional[int] = 10000,
        batch_size:Optional[int] = 32
    ):
        super(VanillaReplayBuffer, self).__init__(obs_dim, action_dim, size)
        self.batch_size = batch_size

    def store_transition(
        self, 
        obs:List[float], 
        action: Union[float, np.ndarray], 
        reward: float, 
        next_obs: List[float], 
        done: bool
    ) -> None:
        # if action is of type float, convert it to a list
        if type(action) == float or type(action) == int:
            action = [action]
        # store a transition (SARS' + is_terminal) in the memory
        self.observations[self.pos] = torch.Tensor(obs)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action)
        self.actions[self.pos] = action #.detach().cpu().numpy()
        self.rewards[self.pos] = torch.Tensor([reward])
        self.next_observations[self.pos] = torch.Tensor(next_obs)
        self.dones[self.pos] = done
        self._incr_mem_cnt()

    def sample(self) -> VanillaReplayBufferSamples:
        """Sample a batch of transitions.

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """
        sample_index = np.random.choice(self.size, self.batch_size)

        sample = VanillaReplayBufferSample
        sample.observations = self.observations[sample_index].detach()
        sample.actions = self.actions[sample_index].detach()
        sample.rewards = self.rewards[sample_index].detach()
        sample.next_observations = self.next_observations[sample_index].detach()
        sample.dones = self.dones[sample_index].detach()

        return sample
    
    def correct(self, state_dim: int, action_dim: int) -> None:
        """Makes the necessary changes for one wants to modify action 
        and observations dimensions

        Args:
            state_dim (int)
            action_dim (int)
        """
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)




# === Replay buffer used by PPO ========================================




class PPOReplayBufferSamples(NamedTuple): # Not sure how I would use it.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    logprob_old: torch.Tensor

PPOReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards',
    'returns',
    'advantages',
    'logprob_old'])

class PPOReplayBuffer(BaseReplayBuffer):
    """I'm making this variant because the discounted reward can't be 
    computed until the episode is over. I therefore need a new way of 
    storing the rewards, with an external episode buffer. Furthermore,
    there are not batches since the whole buffer is used for learning and 
    then emptied out
    """
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        discount_factor:float,
        size:Optional[int] = 200,
        batch_size:Optional[int] = 64,
        critic:Optional[CustomNeuralNetwork] = None,
        gae_lambda:Optional[float] = 0.95,
        normalize_advantages:Optional[bool] = True,
        normalize_returns:Optional[bool] = False
    ):
        super(PPOReplayBuffer, self).__init__(obs_dim, action_dim, size)
        self.ep_pos = 0
        self.returns = [] #np.zeros((self.size,1), dtype=np.float32)
        self.advantages = [] #np.zeros((self.size,1), dtype=np.float32) # GAE stuff
        # TODO : I'll have to adapt the following line for discrete case.
        self.logprob_old = [] #np.zeros((self.size,int(self.action_dim/2)), dtype=np.float32)
        self.ep_buffer = self._init_episode_buffer()
        self.discount_factor = discount_factor
        self.critic = critic # addition to compute the GAE stuff
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns
        self.batch_size = batch_size


    # === Init / reinit / correct functions ============================


    def _init_episode_buffer(self) -> PPOReplayBufferSample:
        episode_buffer = PPOReplayBufferSample
        episode_buffer.observations = [] #np.zeros(
            #(self.size, self.obs_dim), dtype=np.float32)
        episode_buffer.actions = [] #np.zeros((self.size, self.action_dim))
        episode_buffer.rewards = [] #np.zeros((self.size,1), dtype=np.float32)
        episode_buffer.next_observations = [] #np.zeros(
            #(self.size, self.obs_dim), dtype=np.float32)
        episode_buffer.dones = [] #np.zeros((self.size,1), dtype=np.bool)
        episode_buffer.returns = [] #np.zeros((self.size,1), dtype=np.float32)
        episode_buffer.advantages = [] #np.zeros((self.size,1), dtype=np.float32)
        # TODO : I'll have to adapt the following line for discrete case.
        episode_buffer.logprob_old = [] #np.zeros((self.size,int(self.action_dim/2)), dtype=np.float32)
        self.ep_pos = 0
        return episode_buffer

    def erase(self):
        self.observations = [] #np.zeros(
        #    (self.size, self.obs_dim), dtype=np.float32)
        self.actions = [] #np.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = [] #np.zeros((self.size,1), dtype=np.float32)
        self.next_observations = [] #np.zeros(
        #    (self.size, self.obs_dim), dtype=np.float32)
        self.dones = [] #np.zeros((self.size,1), dtype=np.bool)
        self.returns = [] #np.zeros((self.size,1), dtype=np.float32)
        self.advantages = [] #np.zeros((self.size,1), dtype=np.float32)
        # TODO : adapt to discrete action
        self.logprob_old = [] #np.zeros((self.size,int(self.action_dim/2)), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.ep_buffer = self._init_episode_buffer()

    def correct(self, state_dim: int, action_dim: int) -> None:
        """Makes the necessary changes for one wants to modify action 
        and observations dimensions

        Args:
            state_dim (int)
            action_dim (int)
        """
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.observations = [] #np.zeros(
        #    (self.size, self.obs_dim), dtype=np.float32)
        self.actions = [] #np.zeros((self.size, self.action_dim)) # TODO: add type later
        self.next_observations = [] #np.zeros(
        #    (self.size, self.obs_dim), dtype=np.float32)
        # TODO : adapt to discrete case.
        self.logprob_old = [] #np.zeros((self.size,int(self.action_dim/2)), dtype=np.float32)
        self._init_episode_buffer()



    # === Buffer filling functions =====================================



    def store_transition(
        self, 
        obs: List[float], 
        action: Union[List[float],float], 
        reward:float, 
        next_obs: List[float], 
        done:bool,
        logprob: List[float]
    ):
        self._store_in_episode_buffer(obs, action, reward, next_obs, done, logprob)

    def _store_in_episode_buffer(
        self, 
        obs: List[float], 
        action: Union[List[float],float], 
        reward: float, 
        next_obs: List[float], 
        done: bool, 
        logprob: List[float]
    ):
        """stare thetransition in the episode buffer. If the episode is done,
        compute the returns and advantages. Then store everything in the main
        buffer and reinit the episode buffer.
        """
        # store the transition in the episode buffer
        self.ep_buffer.observations += [obs]
        self.ep_buffer.actions += [action]#.detach().cpu().numpy()
        self.ep_buffer.rewards += [reward]
        self.ep_buffer.next_observations += [next_obs]
        self.ep_buffer.dones += [done]
        self.ep_buffer.logprob_old += [logprob]
        
        if done == True:
            # compute advantages for each transition
            #self._compute_advantages_gae_ep()
            self._compute_advantages()
            # send the episode to the main buffer
            self._copy_ep_to_buffer()
            # update the buffer position
            self.pos += self.ep_pos + 1
            # reinitialize the episode buffer
            self.ep_buffer = self._init_episode_buffer()
        else:
            # else, just update the episode buffer position
            self.ep_pos += 1

    def _compute_advantages_gae_ep(self):
        """Compute the advantages for each state in the episode with the 
        GAE method.
        """
        # compute obs value and next obs value for every transition in 
        # the episode buffer
        # the problem I have here is that I assume that my episode will 
        # necessarily be more than one timestep long. It is not the case. 
        # I therefore nee to handle this case.
        obs_values = self.critic(
            self.ep_buffer.observations[:self.ep_pos+1]).detach().squeeze().tolist()
        next_obs_values = self.critic(self.ep_buffer.next_observations[:self.ep_pos+1]
            ).detach().squeeze().tolist()
        
        if type(obs_values) != list:
            obs_values = [obs_values]
            next_obs_values = [next_obs_values]
        
        self.ep_buffer.advantages = [0] * len(self.ep_buffer.next_observations)
        self.ep_buffer.returns = [0] * len(self.ep_buffer.next_observations)
        last_gae_lam = 0
        for step in reversed(range(self.ep_pos+1)): # +1 or not?
            delta: float = self.ep_buffer.rewards[step] + self.discount_factor \
                * next_obs_values[step] * (1.0 - \
                    float(self.ep_buffer.dones[step])) - obs_values[step]
            last_gae_lam = delta + self.discount_factor * self.gae_lambda\
                 * last_gae_lam * (1.0 - float(self.ep_buffer.dones[step]))
            self.ep_buffer.advantages[step] = last_gae_lam
        # compute the returns and store them in the episode buffer.
        self.ep_buffer.returns[:self.ep_pos+1] = [ x + y for x, y in zip(self.ep_buffer.advantages[
            :self.ep_pos+1], obs_values)]
    
    #def compute_advantages(self):
    #    if self.gae_lambda:
        
    def _compute_advantages(self):
        self._compute_ep_returns()
        prev_state_value = self.critic(self.ep_buffer.observations[:self.ep_pos+1])
        self.ep_buffer.advantages[:self.ep_pos+1] = torch.Tensor(self.ep_buffer.returns[:self.ep_pos+1]).unsqueeze(-1) - prev_state_value.detach()
        #self.normalize(advantage) # is it really a good idea?

    def _compute_ep_returns(self):
        """ Compute the returns at each step of the episode and store 
        them in the episode buffer
        """
        #reverse_rewards = torch.flip(self.ep_buffer.rewards[:self.ep_pos+1], (0,1))
        disc_reward:float = 0.0
        self.ep_buffer.returns = [0] * (self.ep_pos +1) # this is bidouillage
        for step in reversed(range(self.ep_pos+1)):
            disc_reward = self.ep_buffer.rewards[step] + self.discount_factor * disc_reward
            self.ep_buffer.returns[step] = disc_reward
        #for i, reward in enumerate(reverse_rewards):
        #    disc_reward = reward + self.discount_factor * disc_reward
        #    self.ep_buffer.returns[self.ep_pos - i] = disc_reward
    
    def _copy_ep_to_buffer(self):
        
        if self.pos + self.ep_pos >= self.size - 1:
            last_val = self.size - self.pos
            self.full = True
            self.observations += self.ep_buffer.observations[:last_val]
            self.actions += self.ep_buffer.actions[:last_val]
            self.rewards += self.ep_buffer.rewards[:last_val]
            self.next_observations += self.ep_buffer.next_observations[:last_val]
            self.dones += self.ep_buffer.dones[:last_val]
            self.returns += self.ep_buffer.returns[:last_val]
            self.advantages += self.ep_buffer.advantages[:last_val]
            self.logprob_old += self.ep_buffer.logprob_old[:last_val]
        else:
            self.observations += self.ep_buffer.observations
            self.actions += self.ep_buffer.actions
            self.rewards += self.ep_buffer.rewards
            self.next_observations += self.ep_buffer.next_observations
            self.dones += self.ep_buffer.dones
            self.returns += self.ep_buffer.returns
            self.advantages += self.ep_buffer.advantages
            self.logprob_old += self.ep_buffer.logprob_old


    def sample(self) -> PPOReplayBufferSamples:
        """Here, all sample are selected

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """
        # make an iterator returning minibatches of size batch_size from sample
        shuf_ids = np.random.permutation(len(self.observations))
        it = iter(range(0, self.size, self.batch_size))
        for i in it:
            sample = PPOReplayBufferSample
            sample.observations = torch.tensor(self.observations, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            sample.actions = torch.tensor(self.actions, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            sample.rewards = torch.tensor(self.rewards, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            sample.next_observations = torch.tensor(self.next_observations, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            sample.dones = torch.tensor(self.dones, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            sample.returns = torch.tensor(self.returns, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            if self.normalize_returns:
                sample.returns = self._normalize(sample.returns)
            sample.advantages = torch.tensor(self.advantages, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            if self.normalize_advantages:
                sample.advantages = self._normalize(sample.advantages)
            sample.logprob_old = torch.tensor(self.logprob_old, dtype=torch.float32)[shuf_ids[i:i+self.batch_size]]
            yield sample

    def _to_tensor(self, sample):
        pass

    
    def _normalize(self, input: torch.Tensor):
        """Normalize the input with the mean and std of the buffer
        """
        mean = input.mean().detach()
        std = input.std().detach()
        ret_val = (input - mean) / (std + 1e-8)
        if torch.isnan(ret_val).any():
            ret_val = input
        return ret_val

    
        
    



# === Hindisght Exprerience Replay Buffer ==============================



class HER(BaseReplayBuffer):
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        discount_factor:float,
        size:Optional[int] = 10000,
        batch_size:Optional[int] = 32
    ):
        super(HER, self).__init__(obs_dim, action_dim, size)
        self.ep_pos = 0
        self.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.ep_buffer = self._init_episode_buffer()
        self.discount_factor = discount_factor
        self.batch_size = batch_size

    # Took the following 3 functions in PPO directly
    
    def store_transition(self, obs, action, reward, next_obs, done):
        self._store_in_episode_buffer(obs, action, reward, next_obs, done)
    
    def _init_episode_buffer(self) -> PPOReplayBufferSample:
        episode_buffer = PPOReplayBufferSample
        episode_buffer.observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        episode_buffer.actions = torch.zeros((self.size, self.action_dim))
        episode_buffer.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        episode_buffer.next_observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        episode_buffer.dones = torch.zeros((self.size,1), dtype=torch.bool)
        episode_buffer.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.ep_pos = 0
        return episode_buffer

    def _store_in_episode_buffer(self, obs, action, reward, next_obs, done):
        # store the transition in the episode buffer
        self.ep_buffer.observations[self.ep_pos] = torch.Tensor(obs)
        self.ep_buffer.actions[self.ep_pos] = action#.detach().cpu().numpy()
        self.ep_buffer.rewards[self.ep_pos] = torch.Tensor([reward])
        self.ep_buffer.next_observations[self.ep_pos] = torch.Tensor(next_obs)
        self.ep_buffer.dones[self.ep_pos] = done
        
        if done == True:
            # if this is the final transition, compute the expected return
            # and store the episode in the main buffer.
            disc_reward = 0.0
            reverse_rewards = self.ep_buffer.rewards[:self.ep_pos+1].flip(1)
            for i, reward in enumerate(reverse_rewards):
                disc_reward = reward + self.discount_factor * disc_reward
                self.ep_buffer.returns[self.ep_pos - i] = disc_reward
                # TODO: add reward function for other objectives ; add the goals
            self._copy_to_buffer()
            self.pos += self.ep_pos + 1
        else:
            self.ep_pos += 1
    
    def _copy_to_buffer(self):
        # for later: I don't need to put values at the beginning of the 
        # buffer when I have too much experience, because the buffer is 
        # going to be emptied out anyway after learning.
        # if the episode is too long, put nly part of it in the buffer.
        # TODO: I suspect that I fucked up the indices. It may be diff 
        # instead of diff + 1
        if self.pos + self.ep_pos >= self.size - 1:
            diff = self.size - self.pos
            self.observations[self.pos:] = self.ep_buffer.observations[:diff+1]
            self.actions[self.pos:] = self.ep_buffer.actions[:diff+1]
            self.rewards[self.pos:] = self.ep_buffer.rewards[:diff+1]
            self.next_observations[self.pos:] =\
                self.ep_buffer.next_observations[:diff+1]
            self.dones[self.pos:] = self.ep_buffer.dones[:diff+1]
            self.returns[self.pos:] = self.ep_buffer.returns[:diff+1]
            self.full = True
        else:
            self.observations[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.observations[:self.ep_pos+1]
            self.actions[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.actions[:self.ep_pos+1]
            self.rewards[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.rewards[:self.ep_pos+1]
            self.next_observations[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.next_observations[:self.ep_pos+1]
            self.dones[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.dones[:self.ep_pos+1]
            self.returns[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.returns[:self.ep_pos+1]



# === Performance testing Exprerience Replay Buffer ====================



class PerfoReplayBuffer(BaseReplayBuffer):
    """This class is only used for performance testing
    i.e. comparison of performance between CPU and GPU for different 
    buffer sizes.
    """
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        size:Optional[int] = 10000,
    ):
        super(PerfoReplayBuffer, self).__init__(obs_dim, action_dim, size)

    def fill(self, obs, action, reward, next_obs, done):
        """ This function fills the buffer with one observation?"""
        self.observations = torch.full(self.size,obs)
        self.actions = torch.full(self.size,action)#.detach().cpu().numpy()  
        self.rewards = torch.full(self.size,[reward])
        self.next_observations = torch.full(self.size, next_obs)
        self.dones = torch.full(self.size, done)

    def noise_init(self, device):
        """ Fills the buffer with noise"""
        self.observations = self.noise_tensor(self.observations.to(device))
        self.actions = torch.unsqueeze(
            torch.randint(self.action_dim, (self.size,), device=device), 1)
        self.rewards = self.noise_tensor(self.rewards.to(device))
        self.next_observations = self.noise_tensor(
            self.next_observations.to(device))
        self.dones = torch.full((self.size,), 0, device=device)
    
    def noise_tensor(self, tensor):
        with torch.no_grad():
            return tensor.uniform_(-1.0, 1.0)


    def sample(self) -> VanillaReplayBufferSamples:
        """Sample whole buffer

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """
        sample = VanillaReplayBufferSample
        sample.observations = self.observations.detach()
        sample.actions = self.actions.detach()
        sample.rewards = self.rewards.detach()
        sample.next_observations = self.next_observations.detach()
        sample.dones = self.dones.detach()

        return sample
    
    def correct(self, state_dim: int, action_dim: int) -> None:
        """Makes the necessary changes when one wants to modify action 
        and observations dimensions

        Args:
            state_dim (int)
            action_dim (int)
        """
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.next_observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)


#%%
import numpy as np
a = np.array([0, 1, 2, 3, 4])
a[:2]
for i in enumerate(a[:3][::-1]):
    print(i)
