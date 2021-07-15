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
        batch_size:Optional[int] = 32
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
    
    def store_transition(self, obs, action, reward, next_obs, done):
        # store a transition (SARS' + is_terminal) in the memory
        self.observations[self.pos] = torch.Tensor(obs)
        self.actions[self.pos] = action#.detach().cpu().numpy()
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = torch.Tensor(next_obs)
        self.dones[self.pos] = done
        self._incr_mem_cnt()
        
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
