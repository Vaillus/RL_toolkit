from other_agents.TDAgent import TDAgent
from DQN.DQNAgent import DQNAgent
from GradientPolicyMethods.REINFORCEAgent import REINFORCEAgent
from GradientPolicyMethods.REINFORCEAgentWithBaseline import REINFORCEAgentWithBaseline
from GradientPolicyMethods.ActorCriticAgent import ActorCriticAgent
from GradientPolicyMethods.PPOAgent import PPOAgent
from GradientPolicyMethods.DDPGAgent import DDPGAgent
from other_agents.AbaddonAgent import AbaddonAgent
from modules.logger import Logger

from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod
import torch
from utils import get_params
import wandb



class AgentInterface(ABC):
    def __init__(
        self, 
        type,
        agent_kwargs, 
        logger: Optional[Logger] = None
    ):
        self.agent = None 
        self.logger = None
        self.type = type
        self.set_logger(logger)
        self.init_agent(agent_kwargs)
    
    def set_logger(self, logger):
        self.logger = logger
        if self.agent is not None:
            self.agent.set_logger(logger)
        
    @abstractmethod
    def init_agent(self, agent_kwargs:dict):
        raise ValueError("This function is not supposed to be accessed because \
             one of the children should be used instead.")

    def _init_single_agent(self, agent_kwargs:Dict[str, Any]):
        """Create and return an agent. The type of agent depends on the 
        self.type parameter
        Args:
            agent_params (dict)

        Returns:
            Agent: the agent initialized
        """
        if self.logger: # TODO: replace with a new function in the logger.
            if self.logger.wandb: 
                agent_kwargs = self.logger.get_config()
        agent = None
        if self.type == "DQN":
            agent = DQNAgent(**agent_kwargs)
        elif self.type == "tile coder test":
            agent = self._init_tc_agent(**agent_kwargs)
        elif self.type == "REINFORCE":
            agent = REINFORCEAgent(**agent_kwargs)
        elif self.type == "REINFORCE with baseline":
            agent = REINFORCEAgentWithBaseline(**agent_kwargs)
        elif self.type == "actor-critic":
            agent = ActorCriticAgent(**agent_kwargs)
        elif self.type == "Abaddon test":
            agent = AbaddonAgent(**agent_kwargs)
        elif self.type == "PPO":
            agent = PPOAgent(**agent_kwargs)
        elif self.type == "DDPG":
            agent = DDPGAgent(**agent_kwargs)
        else:
            raise ValueError(f"agent not initialized because {self.type} is not \
                recognised")
        return agent
    
    def _init_tc_agent(self, agent_params):
        """initialization of a tile coder agent, which depends on the 
        gym environment
        I sould probably get rid of tile coder.
        """
        assert self.environment_name == "gym", "tile coder not supported for godot environments"
        
        agent_params["agent_info"]["function_approximator_info"]["env_min_values"] = \
            self.environment.observation_space.low
        agent_params["agent_info"]["function_approximator_info"]["env_max_values"] = \
            self.environment.observation_space.high
        agent = TDAgent(agent_params)
         
        return agent
      
    def _get_single_agent_action(
        self, 
        state_data, 
        reward_data=None, 
        start=False,
        tot_timestep=0
    ):
        """if this is the first state of the episode, get the first 
        action of the agent else, also give reward of the previous 
        action to complete the previous transition.

        Args:
            agent (Agent)
            state_data (dict)
            reward_data (dict, optional): Defaults to None.
            start (bool, optional): indicates whether it is the first 
                                    step of the agent. Defaults to False.

        Returns:
            int : id of action taken
        """
        if start is True:
            if self.type == "DQN":
                action_data = self.agent.start(state_data, tot_timestep)
            else:
                action_data = self.agent.start(state_data)
        else:
            if self.type == "DQN":
                action_data = self.agent.step(state_data, reward_data, tot_timestep)
            else:
                action_data = self.agent.step(state_data, reward_data)
        return action_data
    
    def get_state_value_eval(self, state):
        return self.agent.get_state_value_eval(state)
    def get_action_value_eval(self, state:torch.Tensor):
        return self.agent.get_action_value_eval(state)
    
    def get_action_values_eval(self, state:torch.Tensor, actions:torch.Tensor):
        return self.agent.get_action_values_eval(state, actions)
    
    def get_discount(self):
        return self.agent.get_discount()
        
    
    
class SingleAgentInterface(AgentInterface):
    def __init__(
        self,
        type,
        agent_kwargs,
        seed,
        logger
    ):
        super(SingleAgentInterface, self).__init__(type, agent_kwargs, logger)
        self.set_seed(seed)
    
    def init_agent(self, agent_params):
        self.agent = self._init_single_agent(agent_params)
    
    def set_seed(self, seed):
        self.agent.set_seed(seed)

    def get_action(self, state_data, reward_data=None, start=False,
        tot_timestep: Optional[int] = 0):

        action_data = self._get_single_agent_action(
            state_data, reward_data, start, tot_timestep)

        return action_data

    def end(self, state_data, reward_data):
        self.agent.end(state_data, reward_data)

    def learn_from_experience(self):
        self.agent.learn_from_experience()
    
    def check(self, action_type:str, action_dim:int, state_dim:int) -> bool:
        """Check if environment and agent action types and dimensions match

        Args:
            action_type (str): environment action type
            action_dim (int): environment action dimension
            state_dim (int): environment state dimension

        Returns:
            bool: is the agent compatible with the environment?
        """
        agent_action_type = get_params("misc/agent_action_type")
        assert action_type == agent_action_type[self.type], "env and agent\
         action types don't match"
        is_ok = self.agent.num_actions == action_dim and self.agent.state_dim == state_dim
        return is_ok
    
    def fix(self, action_dim:int, state_dim:int):
        self.agent.adjust_dims(state_dim, action_dim)
    


class MultiAgentInterface(AgentInterface):
    def __init__(
        self,
        type,
        agent_names,
        agent_kwargs,
        seed,
        logger
    ):
        super(MultiAgentInterface, self).__init__(type, agent_kwargs, logger)
        self.set_seed(seed)
        self.agents_names = agent_names

    def init_agent(self, agent_params):
        self.agents_names = self.environment.agent_names
        self.agent = {}
        for agent_name in self.agents_names:
            self.agent[agent_name] = self._init_single_agent(agent_params)
        
    def set_seed(self, seed):
        for agent_name in self.agents_names:
            self.agent[agent_name].set_seed(seed)
    
    def set_logger(self, logger):
        self.logger = logger
        for agent_name in self.agents_names:
            self.agent[agent_name].set_logger(logger)

    def get_action(
        self, 
        state_data: List[dict],
        reward_data: Optional[List[dict]] = [],
        start: Optional[bool] = False,
        tot_timestep: Optional[int] = 0
    ):
        """ distribute states to all agents and get their actions back.

        Args:
            state_data (dict)
            reward_data (dict, optional): Defaults to None.
            start (bool, optional): indicates whether it is the first 
                step of the agent. Defaults to False.

        Returns:
            dict
        """
        if not start:
            assert len(reward_data) == 0, "If it's not the first state, \
                there should be a reward data"
        action_data = []
        # Isolate state and reward data for each agent and get their actions
        # individually.
        for n_agent in enumerate(state_data):
            agent_name = state_data[n_agent]["name"]
            agent_state = state_data[n_agent]["state"]
            agent_reward = None
            if not start:
                agent_reward = reward_data[n_agent]['reward']
            action = self._get_single_agent_action(agent=self.agent[agent_name], 
                                                    state_data=agent_state, 
                                                    reward_data=agent_reward, 
                                                    start=start,
                                                    tot_timestep=tot_timestep)
            action_data.append({"name": agent_name, "action": action})
        return action_data
    
    def end(self, state_data, reward_data):
        """send the terminal state and the final reward to every agent 
        so they can complete their last transitions

        Args:
            state_data (dict)
            reward_data (dict)
        """
        for n_agent in enumerate(state_data):
                agent_name = state_data[n_agent]["name"]
                agent_state = state_data[n_agent]["state"]
                agent_reward = reward_data[n_agent]["reward"]

                self.agent[agent_name].end(agent_state, agent_reward)
    
   
