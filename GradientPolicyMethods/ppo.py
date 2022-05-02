from custom_nn import CustomNeuralNetwork
from policy_nn import PolicyNetwork
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, Type, Dict, Any, List, Union
from modules.replay_buffer import PPOReplayBuffer
from modules.logger import Logger
from utils import set_random_seed

from modules.curiosity import Curiosity
import numpy as np

import cProfile
import pstats


MSELoss = torch.nn.MSELoss()

class PPOAgent:
    def __init__(
        self, 
        policy_estimator_info: Dict[str, Any],
        function_approximator_info: Dict[str, Any],
        memory_info: Dict[str, Any],
        curiosity_info: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 0,
        discount_factor: Optional[float] = 0.9,
        num_actions: Optional[int] = 1,
        is_continuous: bool = False,
        state_dim: Optional[int] = 1, 
        clip_range: Optional[float] = 0.2,
        value_coeff: Optional[float] = 1.0,
        entropy_coeff: Optional[float] = 0.01,
        n_epochs: Optional[int] = 8,
        gae_lambda: Optional[float] = 0.95,
        normalize_advantages: Optional[bool] = True,
        grad_clip_range: Optional[float] = 0.5
    ):
        self.γ = discount_factor
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.is_continuous = is_continuous

        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.grad_clip_range = grad_clip_range
        self.actor: CustomNeuralNetwork =\
            self.initialize_policy_estimator(policy_estimator_info)
        self.critic: CustomNeuralNetwork =\
            self.initialize_function_approximator(function_approximator_info)
        nn.init.orthogonal_(self.critic.layers[-1].weight.data, 1.0)
        # make it optional ?
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_range)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_range)
        self.replay_buffer = self.init_memory_buffer(memory_info)

        self.curiosity = self.init_curiosity(curiosity_info)

        self.seed = seed
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs

        self.previous_state: List[float] = None
        self.previous_action = None

        self.logger = None
        # metrics to be logged
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.entropies = []
        self.intrinsic_rewards = []
        self.approx_kls = []
        self.explained_vars = []

        # self.memory_size = memory_info.get("size", 200) #why?



    # ====== Initialization functions ==================================



    def initialize_policy_estimator(self, params: Dict) -> CustomNeuralNetwork:
        params["is_continuous"] = self.is_continuous
        return PolicyNetwork(
            **params, 
            input_dim=self.state_dim, 
            output_dim=self.num_actions
        )

    def initialize_function_approximator(self, params: Dict) -> CustomNeuralNetwork:
        return CustomNeuralNetwork(
            **params, 
            input_dim=self.state_dim, 
            output_dim=1
        )
    
    def init_memory_buffer(self, params: Dict) -> PPOReplayBuffer:
        params["obs_dim"] = self.state_dim
        params["action_dim"] = self.num_actions
        params["discount_factor"] = self.γ
        params["critic"] = self.critic # to compute the values of the states
        params["gae_lambda"] = self.gae_lambda
        params["normalize_advantages"] = self.normalize_advantages
        return PPOReplayBuffer(**params)

    def init_curiosity(self, params: Dict) -> Curiosity:
        if params:
            return Curiosity(**params)
        else:
            return Curiosity()
    
    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)
            self.critic.set_seed(seed)
    
    def set_logger(self, logger:Type[Logger]):
        self.logger = logger
        self.logger.wandb_watch([self.actor, self.critic], type="grad")

    def control(self):
        if self.replay_buffer.full:
            """if self.is_continuous:
                old_log_probs = self.actor_cont(batch.observations
                ).log_prob(batch.actions).detach()
            else:
                probs_old = self.actor(batch.observations).detach()
            """
            for _ in range(self.n_epochs):
                for batch in self.replay_buffer.sample():
                    # === critic stuff
                    obs_values = self.critic(batch.observations)
                    obs_values = torch.squeeze(obs_values)
                    assert obs_values.shape == batch.rewards.shape, "something \
                        wrong with the shapes of the tensors for the computation of value loss"
                    value_loss = MSELoss(obs_values, batch.returns)
                    self.critic.backpropagate(value_loss * self.value_coeff)
                    self.value_losses.append(value_loss.item())
                    # explained variance as a new metric. 
                    # measures how much the value of the state is correctly predicted.
                    y_pred, y_true = obs_values.detach().numpy(), batch.returns.detach().numpy() 
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    self.explained_vars.append(explained_var)

                    # === actor stuff
                    # TODO: sample minibatches here and iterate over them.
                    # nb: needed only when the batch size is too high for the cpu/gpu. 
                    # Not the case here.
                    #probs_new = self.get_proba(batch.observations, batch.actions)
                    if self.is_continuous:
                        # get the probabilities of the action taken with the updated policy
                        log_probs = \
                            self.actor_cont(batch.observations, plot=True).log_prob(batch.actions.unsqueeze(-1))
                        # compute the importance-sampling ratio for each action and clip them 
                        log_ratio = (log_probs.squeeze() - batch.logprob_old.squeeze())
                        assert log_probs.squeeze().shape == batch.logprob_old.squeeze().shape, "something \
                            wrong with the shapes of the tensors for the computation of policy loss"
                        ratio = torch.exp(log_ratio)
                        # compute metric to measure aggressivity of policy change
                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - log_ratio).mean()
                        self.approx_kls.append(approx_kl.item())
                        clipped_ratio = torch.clamp(
                            ratio, 
                            min = 1 - self.clip_range, 
                            max = 1 + self.clip_range
                        ) # OK
                        # recompute advantage here. GAE(lambda), then.
                        policy_loss = torch.min(
                            batch.advantages.detach() * ratio, 
                            batch.advantages.detach() * clipped_ratio
                        ) # OK
                        policy_loss = - policy_loss.mean() # OK
                        
                        entropy = self.actor_cont(batch.observations).entropy().mean()
                        self.entropies.append(entropy.item())
                        entropy_loss = - entropy * self.entropy_coeff
                        
                        self.actor.backpropagate(policy_loss + entropy_loss)

                        self.entropy_losses.append(entropy_loss.item())
                        self.policy_losses.append(policy_loss.item())
                    else:
                        # get the probabilities of the action taken with the updated policy
                        probs_new = self.actor(batch.observations)
                        # compute the importance-sampling ratio for each action and clip them 
                        ratio = torch.exp(torch.log(probs_new) - torch.log(probs_old)) # ok
                        ratio = torch.gather(ratio, 1, batch.actions.long()) #ok 
                        clipped_ratio = torch.clamp(
                            ratio, 
                            min = 1 - self.clip_range, 
                            max = 1 + self.clip_range
                        ) # OK
                        policy_loss = torch.min(
                            batch.advantages.detach() * ratio, 
                            batch.advantages.detach() * clipped_ratio
                        ) # OK
                        policy_loss = - policy_loss.mean() # OK
                        self.policy_losses.append(policy_loss.item())
                        
                        entropy = -(
                            torch.sum(
                                probs_new * torch.log(probs_new), dim=1, keepdim=True
                            ).mean()
                        )
                        self.entropies.append(entropy.item())
                        entropy_loss = - entropy * self.entropy_coeff
                        self.entropy_losses.append(entropy_loss.item())

                        # computing ICM loss
                        #intrinsic_reward = self.compute_icm_loss(batch, self.actor)
                        #intrinsic_rewards.append(intrinsic_reward.item())
                        
                        self.actor.backpropagate(policy_loss + entropy_loss) #+ intrinsic_reward)
            self._log_control_metrics()
            # the replay buffer is used only one (completely) and then 
            # emptied out
            self.replay_buffer.erase()
    
    def _log_control_metrics(self):
        if self.is_continuous:
            self.logger.log({
                "Agent info/ approx KL": np.mean(self.approx_kls)},
                type="agent"
            )
        self.logger.log({
            'Agent info/critic loss': np.mean(self.value_losses),
            'Agent info/actor loss': np.mean(self.policy_losses),
            'Agent info/entropy': np.mean(self.entropies),
            'Agent info/entropy loss': np.mean(self.entropy_losses),
            'Agent info/intrinsic reward': np.mean(self.intrinsic_rewards),
            "Agent info/explained variance": np.mean(self.explained_vars)},
            type= "agent")
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.entropies = []
        self.intrinsic_rewards = []
        self.approx_kls = []
        self.explained_vars = []
        

    def get_proba(self, obs, actions):
        if self.is_continuous:
            return 10 ** self.actor_cont(obs).log_prob(actions)
        else:
            return self.actor(obs)

    def compute_icm_loss(self, batch, actor):
        return self.curiosity.compute_icm_loss(batch=batch, nn=actor)



    # ====== Action choice related functions ===========================



    def choose_action(self, state: np.ndarray) -> Union[int, float]:
        if self.is_continuous:  
            action = torch.tanh(self.actor_cont(state).sample())
            #action = self.actor_cont(state).sample().clamp(-1, 1)
            #action_chosen = torch.normal(mu, std).clamp(-1, 1)
            return action.item()
        else:
            action_probs = Categorical(self.actor(state))
            action_chosen = action_probs.sample()
            return action_chosen.item()



    # ====== Agent core functions ======================================



    def start(self, state: np.ndarray):
        # choosing the action to take
        current_action = self.choose_action(state) 

        self.previous_action = current_action
        self.previous_state = state.tolist()

        return current_action

    def step(self, state: np.ndarray, reward: float):

        #intrinsic_reward = self.curiosity.get_intrinsic_reward(self.actor, self.previous_state, state, self.previous_action)
        #reward += intrinsic_reward
        # storing the transition in the memory for further use
        # TODO : compute that elsewhere
        logprob = torch.distributions.Normal(
            *self.actor(self.previous_state)).log_prob(
                torch.tensor(self.previous_action)).detach().tolist()
        self.replay_buffer.store_transition(
            self.previous_state, 
            self.previous_action, 
            reward, 
            state.tolist(), 
            False, 
            logprob
        )
        # getting the action values from the function approximator
        current_action = self.choose_action(state)
        self.control()

        self.previous_action = current_action
        self.previous_state = state.tolist()

        return current_action

    def end(self, state: List, reward:float):
        """ receive the terminal state and reward to stor the final transition. 
        Apparently I also compute the advantage at the end of the episode... """
        #intrinsic_reward = self.curiosity.get_intrinsic_reward(self.actor, self.previous_state, state, self.previous_action)
        #reward += intrinsic_reward
        # storing the transition in the function approximator memory for further use
        
        # measure time spent executing the next line with cstats and print it
        #with cProfile.Profile() as pr:
        logprob = torch.distributions.Normal(
            *self.actor(self.previous_state)).log_prob(
                torch.tensor(self.previous_action)).detach().tolist()
        self.replay_buffer.store_transition(
            self.previous_state, 
            self.previous_action, 
            reward, #+ intrinsic_reward
            state.tolist(), 
            True, 
            logprob
        )
        #stats = pstats.Stats(pr)
        #self.compute_ep_advantages()
        self.control()

    def get_state_value_eval(self, state:np.ndarray):
        """ Used in the probe environmnents to test the agent."""
        if self.num_actions > 1:
            state_value = self.actor(state).data
        else: 
            state_value = self.critic(state).data
        return state_value

    def adjust_dims(self, state_dim:int, action_dim:int):
        """ Called when the agent dimension doesn't fit the environment 
        dimension. Reinitialize some parts of the agent so they fit the 
        environment.
        """
        self.state_dim = state_dim
        self.num_actions = action_dim

        self.actor.reinit_layers(state_dim, action_dim)
        self.critic.reinit_layers(state_dim, 1)
        self.replay_buffer.correct(state_dim, action_dim)
        self.logger.wandb_watch([self.actor, self.critic], type="grad")

    def compute_ep_advantages(self):
        """compute the GAE advantages for the episode buffer"""
        self.replay_buffer._compute_advantages_gae()

    def actor_cont(self, state: np.ndarray, plot: bool = False):
        """ Return the action distribution for the given state in the 
        continuous case."""
        mu, sigma = self.actor(state)
        if plot:
            self.logger.log({
                "Agent info/sigma": sigma.mean().item(),
                "Agent info/mu": mu.mean().item()}, log_freq=500)
        
        action_probs = torch.distributions.Normal(mu, sigma)
        return action_probs
        
    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        """ for plotting purposes only. Called from continuous probe environment. 
        """
        action_probs = torch.exp(self.actor_cont(state).log_prob(actions))
        return action_probs.detach().numpy()