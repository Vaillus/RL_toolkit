import numpy as np
import torch
from typing import NamedTuple, Optional
from collections import namedtuple

#from torch._C import float32

class ReplayBuffer:
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
        self.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.size,1), dtype=torch.bool)
    
    def _incr_mem_cnt(self) -> None:
        """ Increment the memory counter and resets it to 0 when reached 
        the memory size value to avoid a too large value
        """
        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0


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

class VanillaReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        size:Optional[int] = 10000,
        batch_size:Optional[int] = 32
    ):
        super(VanillaReplayBuffer, self).__init__(obs_dim, action_dim, size)
        self.batch_size = batch_size

    def store_transition(self, obs, action, reward, next_obs, done):
        # store a transition (SARS' + is_terminal) in the memory
        self.observations[self.pos] = torch.Tensor(obs)
        self.actions[self.pos] = action#.detach().cpu().numpy()
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




class PPOReplayBufferSamples(NamedTuple): # Not sure how I would use it.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor

PPOReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards',
    'returns'])

class PPOReplayBuffer(ReplayBuffer):
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
        size:Optional[int] = 200
    ):
        super(PPOReplayBuffer, self).__init__(obs_dim, action_dim, size)
        self.ep_pos = 0
        self.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.ep_buffer = self._init_episode_buffer()
        self.discount_factor = discount_factor

    def store_transition(self, obs, action, reward, next_obs, done):
        self._store_in_episode_buffer(obs, action, reward, next_obs, done)

    def sample(self) -> PPOReplayBufferSamples:
        """Here, all sample are selected

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """

        sample = PPOReplayBufferSample
        sample.observations = self.observations.detach()
        sample.actions = self.actions.detach()
        sample.rewards = self.rewards.detach()
        sample.next_observations = self.next_observations.detach()
        sample.dones = self.dones.detach()
        sample.disc_rewards = self.returns.detach()
        self.erase()

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
        self._init_episode_buffer()

    def _init_episode_buffer(self) -> PPOReplayBufferSample:
        episode_buffer = PPOReplayBufferSample
        episode_buffer.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        episode_buffer.actions = torch.zeros((self.size, self.action_dim))
        episode_buffer.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        episode_buffer.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
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
            self._copy_to_buffer()
            self.pos += self.ep_pos + 1
            
        else:
            self.ep_pos += 1

    def _copy_to_buffer(self):
        # for later: I don't need to put values at the beginning of the 
        # buffer when I have too much experience, because the buffer is 
        # going to be emptied out anyway after learning.
        # TODO: I suspect that I fucked up with the indices. It should be diff 
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

    def erase(self):
        self.episode_buffer = self._init_episode_buffer()
        self.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.size,1), dtype=torch.bool)
        self.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.pos = 0
        self.full = False



class HER(ReplayBuffer):
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
        episode_buffer.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        episode_buffer.actions = torch.zeros((self.size, self.action_dim))
        episode_buffer.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        episode_buffer.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
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
            self._copy_to_buffer()
            self.pos += self.ep_pos + 1
            
        else:
            self.ep_pos += 1
    
    def _copy_to_buffer(self):
        # for later: I don't need to put values at the beginning of the 
        # buffer when I have too much experience, because the buffer is 
        # going to be emptied out anyway after learning.
        # if the episode is too long, put nly part of it in the buffer.
        # TODO: I suspect that I fucked up with the indices. It should be diff 
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




#%%
import numpy as np
a = np.array([0, 1, 2, 3, 4])
a[:2]
for i in enumerate(a[:3][::-1]):
    print(i)
