import numpy as np
import torch
from typing import NamedTuple, Optional
from collections import namedtuple

from CustomNeuralNetwork import CustomNeuralNetwork

#from torch._C import float32

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



# === Replay buffer used by PPO ========================================



class PPOReplayBufferSamples(NamedTuple): # Not sure how I would use it.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

PPOReplayBufferSample = namedtuple('ReplayBufferSample', [
    'observations',
    'actions',
    'next_observations',
    'dones',
    'rewards',
    'returns',
    'advantages'])

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
        gae_lambda:Optional[float] = 0.95
    ):
        super(PPOReplayBuffer, self).__init__(obs_dim, action_dim, size)
        self.ep_pos = 0
        self.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.advantages = torch.zeros((self.size,1), dtype=torch.float32) # GAE stuff
        self.ep_buffer = self._init_episode_buffer()
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.critic = critic # addition to compute the GAE stuff
        self.gae_lambda = gae_lambda

    def store_transition(self, obs, action, reward, next_obs, done):
        self._store_in_episode_buffer(obs, action, reward, next_obs, done)

    def sample(self) -> PPOReplayBufferSamples:
        """Here, all sample are selected

        Returns:
            ReplayBufferSamples: observations, actions, rewards, 
            next_observations, dones
        """

        # TODO: compute advantage and returns
        # TODO: make mini batches
        # TODO: return an iterator

        sample = PPOReplayBufferSample
        sample.observations = self.observations.detach()
        sample.actions = self.actions.detach()
        sample.rewards = self.rewards.detach()
        sample.next_observations = self.next_observations.detach()
        sample.dones = self.dones.detach()
        sample.returns = self.returns.detach()
        sample.advantages = self.advantages
        #self.episode_buffer = self._init_episode_buffer() 
        # # this deletes what is inside the sample, and nothing is learned.

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
        self.observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.next_observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        self._init_episode_buffer()

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
        episode_buffer.advantages = torch.zeros((self.size,1), dtype=torch.float32)
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
            # compute advantages for each transition
            self._compute_advantages_gae()
            # send the episode to the main buffer
            self._copy_ep_to_buffer()
            # update the buffer position
            self.pos += self.ep_pos + 1
            # reinitialize the episode buffer
            self.episode_buffer = self._init_episode_buffer()
        else:
            # else, just update the episode buffer position
            self.ep_pos += 1
    
    def _compute_advantages_gae(self):
        # TODO: test the time of execution
        prev_state_value = self.critic(
            self.ep_buffer.observations[:self.ep_pos+1])
        # sample the value of the action chosen in the previous state

        disc_vec = torch.Tensor([
            self.discount_factor ** i for i in range(self.ep_pos+1)])
        k_step_adv = torch.zeros(
            (self.ep_pos+1, self.ep_pos+1), dtype=torch.float32)
        # first step of GAE computation: compute the k-step estimates of 
        # the advantage for each possible k, for each state
        # t is the intex for the state we compute the advantage for,
        # and n is the index for the horizon of the advantage estimate 
        for t in range(self.ep_pos+1):
            for k in range(t, self.ep_pos+1):
                k_step_adv[t, k] = self._compute_k_step_adv(
                    t, k, disc_vec, prev_state_value)
        # for each state, combine the k-step estimates of the advantage 
        # into one single advantage estimate.
        for t in range(self.ep_pos+1):
            # get the advantages vector for the currrent t
            adv_vec = k_step_adv[t, t:] 
            for i in range(adv_vec.shape[0]):
                self.ep_buffer.advantages[t] += adv_vec[i] * self.gae_lambda ** i
    
    def _compute_k_step_adv(
        self, 
        t:int, 
        k:int, 
        disc_vec:torch.Tensor, 
        prev_state_value:torch.Tensor
    ) -> torch.Tensor:
        # compute the k-step estimate of the advantage at state t
        # compute the discounted reward from t to t+k.
        disc_rew = self.ep_buffer.rewards[t:k+1].detach().clone()
        disc_rew *= disc_vec[:disc_rew.shape[0]].unsqueeze(-1)
        disc_rew = sum(disc_rew)

        #if k goes until the end, add the value to the vector of episode returns.
        if k == self.ep_pos:
            self.ep_buffer.returns[t] = disc_rew

        # compute the state value at t+k
        last_state_value = self.critic(self.ep_buffer.next_observations[k]).detach() # detach? not sure
        # combine them in the k-step estimate of the advantage at time t
        adv = - prev_state_value[t] + disc_rew + last_state_value
        return adv
    
    def _compute_advantages(self):
        self._compute_ep_returns()
        prev_state_value = self.critic(self.ep_buffer.observations[:self.ep_pos+1])
        self.ep_buffer.advantages = self.ep_buffer.returns - prev_state_value.detach()
        #self.normalize(advantage) # is it really a good idea?

    def _compute_ep_returns(self):
        """ Compute the returns at each step of the episode and store 
        them in the episode buffer
        """
        reverse_rewards = torch.flip(self.ep_buffer.rewards[:self.ep_pos+1], (0,1))
        for i, reward in enumerate(reverse_rewards):
            disc_reward = reward + self.discount_factor * disc_reward
            self.ep_buffer.returns[self.ep_pos - i] = disc_reward
        

    def _copy_ep_to_buffer(self):
        # Might be diff+1 instead of diff? diff+1 did'nt work before.
        # when there is more experience than needed, just cut the 
        # experience and dump the rest.
         # TODO: advantage must be computed before this step because 
        # if the end of the episode is missing, there might be a problem
        #  in the case the reward is only located at the end.

        # in the case the episode length is larger than the remaining 
        # size in the memory, only add the beginning of the episode.
        if self.pos + self.ep_pos >= self.size - 1:
            diff = self.size - self.pos
            self.observations[self.pos:] = self.ep_buffer.observations[:diff]
            self.actions[self.pos:] = self.ep_buffer.actions[:diff]
            self.rewards[self.pos:] = self.ep_buffer.rewards[:diff]
            self.next_observations[self.pos:] =\
                self.ep_buffer.next_observations[:diff]
            self.dones[self.pos:] = self.ep_buffer.dones[:diff]
            self.returns[self.pos:] = self.ep_buffer.returns[:diff]
            self.advantages[self.pos:] = self.ep_buffer.advantages[:diff]
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
            self.advantages[self.pos: self.pos + self.ep_pos+1] =\
                self.ep_buffer.advantages[:self.ep_pos+1]

    def erase(self):
        self.observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.size, self.action_dim)) # TODO: add type later
        self.rewards = torch.zeros((self.size,1), dtype=torch.float32)
        self.next_observations = torch.zeros(
            (self.size, self.obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((self.size,1), dtype=torch.bool)
        self.returns = torch.zeros((self.size,1), dtype=torch.float32)
        self.advantages = torch.zeros((self.size,1), dtype=torch.float32)
        self.pos = 0
        self.full = False



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
