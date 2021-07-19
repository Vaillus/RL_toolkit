import numpy as np
import torch
from typing import NamedTuple, Optional
from collections import namedtuple

#from torch._C import float32


class ReplayBufferSamples(NamedTuple): # Not sure how I would use it.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

# TODO: try to replace namedtuple with ReplayBufferSamples
ReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards'])

class ReplayBuffer:
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        size:Optional[int] = 10000,
        batch_size:Optional[int] = 32,
        disc_rew: Optional[bool] = False
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = size
        self.batch_size = batch_size
        self.pos = 0
        self.full = False
        self.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.size,1), dtype=torch.bool)

    def store_transition(self, obs, action, reward, next_obs, done):
        # store a transition (SARS' + is_terminal) in the memory
        self.observations[self.pos] = torch.Tensor(obs)
        self.actions[self.pos] = action#.detach().cpu().numpy()
        self.rewards[self.pos] = torch.Tensor([reward])
        self.next_observations[self.pos] = torch.Tensor(next_obs)
        self.dones[self.pos] = done
        self._incr_mem_cnt()

    def sample(self) -> ReplayBufferSamples:
        """Sample a batch of transitions.

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """
        sample_index = np.random.choice(self.size, self.batch_size)

        sample = ReplayBufferSample
        sample.observations = self.observations[sample_index].detach()
        sample.actions = self.actions[sample_index].detach()
        sample.rewards = self.rewards[sample_index].detach()
        sample.next_observations = self.next_observations[sample_index].detach()
        sample.dones = self.dones[sample_index].detach()

        return sample
        
    def _incr_mem_cnt(self) -> None:
        """ Increment the memory counter and resets it to 0 when reached 
        the memory size value to avoid a too large value
        """
        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0
    
    def reinit(self, state_dim: int, action_dim: int) -> None:
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






PPOReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards',
    'disc_rewards'])

class PPOReplayBuffer:
    """I'm making this variant because the discounted reward can't be 
    computed until the episode is over. I therefore need a new way of 
    storing the rewards, with an external episode buffer.
    """
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        discount_factor:float,
        size:Optional[int] = 10000,
        batch_size:Optional[int] = 32
        
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = size
        self.batch_size = batch_size
        self.pos = 0
        self.ep_pos = 0
        self.full = False
        self.observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.next_observations = torch.zeros((self.size, self.obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.size,1), dtype=torch.bool)
        self.disc_rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.ep_buffer = self._init_episode_buffer()
        self.discount_factor = discount_factor

    def store_transition(self, obs, action, reward, next_obs, done):
        self._store_in_episode_buffer(obs, action, reward, next_obs, done)
        # store a transition (SARS' + is_terminal) in the memory
        self.observations[self.pos] = torch.Tensor(obs)
        self.actions[self.pos] = action#.detach().cpu().numpy()
        self.rewards[self.pos] = torch.Tensor([reward])
        self.next_observations[self.pos] = torch.Tensor(next_obs)
        self.dones[self.pos] = done
        #self._incr_mem_cnt()

    def sample(self) -> ReplayBufferSamples:
        """Sample a batch of transitions.

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """
        sample_index = np.random.choice(self.size, self.batch_size)

        sample = ReplayBufferSample
        sample.observations = self.observations[sample_index].detach()
        sample.actions = self.actions[sample_index].detach()
        sample.rewards = self.rewards[sample_index].detach()
        sample.next_observations = self.next_observations[sample_index].detach()
        sample.dones = self.dones[sample_index].detach()

        return sample

    def _incr_mem_cnt(self) -> None:
        """ Increment the memory counter and resets it to 0 when reached 
        the memory size value to avoid a too large value
        """
        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0
    
    def reinit(self, state_dim: int, action_dim: int) -> None:
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
        episode_buffer.disc_rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.ep_pos = 0
        return episode_buffer

    def _store_in_episode_buffer(self, obs, action, reward, next_obs, done):
        self.ep_buffer.observations[self.pos] = torch.Tensor(obs)
        self.ep_buffer.actions[self.pos] = action#.detach().cpu().numpy()
        self.ep_buffer.rewards[self.pos] = torch.Tensor([reward])
        self.ep_buffer.next_observations[self.pos] = torch.Tensor(next_obs)
        self.ep_buffer.dones[self.pos] = done
         
        if done == True:
            # chopé sur PPO.
            #batch_discounted_reward = torch.tensor(np.zeros((self.replay_buffer.size, 1))).float()
            disc_reward = 0.0
            reverse_rewards = self.ep_buffer.rewards[:self.ep_pos+1].flip(1)
            for i, reward in enumerate(reverse_rewards):
                disc_reward = reward + self.discount_factor * disc_reward
                self.ep_buffer.disc_rewards[self.ep_pos - i] = disc_reward
            self._copy_to_buffer()
            self.pos += self.ep_pos #+1?
            
        else:
            self.ep_pos += 1
            if self.ep_pos == self.size:
                raise ValueError(" the episode length is longer than the buffer \
                    length, adjust it.")

    def _copy_to_buffer(self):
        # TODO: check the values and add the fields
        if self.pos + self.ep_pos >= self.size:
            diff = self.size - self.pos
            diff2 = self.ep_pos - diff

            self.disc_rewards[self.pos:] =\
                self.ep_buffer.disc_rewards[:diff+1]
            
            self.disc_rewards[:diff2] =\
                self.ep_buffer.disc_rewards[diff+1:self.ep_pos+1]
        else:
            self.disc_rewards[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.disc_rewards[:self.ep_pos+1] # OK



#%%
import numpy as np
a = np.array([0, 1, 2, 3, 4])
a[:2]
for i in enumerate(a[:3][::-1]):
    print(i)
# %%
