from TDAgent import *
from DQN.DQNAgent import *
from GradientPolicyMethods.REINFORCEAgent import *
from GradientPolicyMethods.REINFORCEAgentWithBaseline import *
from GradientPolicyMethods.ActorCriticAgent import *
from GradientPolicyMethods.PPOAgent import *
from GradientPolicyMethods.DDPGAgent import *
from AbaddonAgent import *

from typing import List, Optional
from abc import ABC, abstractmethod

agent_action_type = {
    "DQN" : "discrete",
    "tile coder test" : "discrete",
    "REINFORCE" : "discrete",
    "REINFORCE with baseline" : "discrete",
    "actor-critic" : "discrete",
    "Abaddon test" : "continuous",
    "PPO" : "discrete",
    "DDPG" : "continuous"
}

class AgentInterface(ABC):
    def __init__(
        self, 
        type,
        agent_kwargs
    ):
        self.agent = None 
        self.type = type
        self.init_agent(agent_kwargs)
        
    @abstractmethod
    def init_agent(self, agent_kwargs:dict):
        raise ValueError("This function is not supposed to be accessed because \
             one of the children should be used instead.")

    def _init_single_agent(self, agent_kwargs):
        """Create and return an agent. The type of agent depends on the 
        self.type parameter
        Args:
            agent_params (dict)

        Returns:
            Agent: the agent initialized
        """
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
        
        params["agent_info"]["function_approximator_info"]["env_min_values"] = \
            self.environment.observation_space.low
        params["agent_info"]["function_approximator_info"]["env_max_values"] = \
            self.environment.observation_space.high
        agent = TDAgent(agent_params)
         
        return agent
      
    def _get_single_agent_action(self, state_data, reward_data=None, start=False):
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
            action_data = self.agent.start(state_data)
        else:
            action_data = self.agent.step(state_data, reward_data)
        return action_data
        
    
    
class SingleAgentInterface(AgentInterface):
    def __init__(
        self,
        type,
        agent_kwargs,
        seed
    ):
        super(SingleAgentInterface, self).__init__(type, agent_kwargs)
        self.set_seed(seed)
    
    def init_agent(self, agent_params):
        self.agent = self._init_single_agent(agent_params)
    
    def set_seed(self, seed):
        self.agent.set_seed(seed)

    def get_action(self, state_data, reward_data=None, start=False):

        action_data = self._get_single_agent_action(
            state_data, reward_data, start)

        return action_data

    def end(self, state_data, reward_data):
        self.agent.end(state_data, reward_data)

    def learn_from_experience(self):
        self.agent.learn_from_experience()
    


class MultiAgentInterface(AgentInterface):
    def __init__(
        self,
        type,
        agent_names,
        agent_kwargs,
        seed
    ):
        super(MultiAgentInterface, self).__init__(type, agent_kwargs)
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

    def get_action(
        self, 
        state_data: List[dict],
        reward_data: List[dict] = [],
        start: Optional[bool] = False
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
        for n_agent in range(len(state_data)):
            agent_name = state_data[n_agent]["name"]
            agent_state = state_data[n_agent]["state"]
            agent_reward = None
            if not start:
                agent_reward = reward_data[n_agent]['reward']
            action = self._get_single_agent_action(agent=self.agent[agent_name], 
                                                    state_data=agent_state, 
                                                    reward_data=agent_reward, 
                                                    start=start)
            action_data.append({"name": agent_name, "action": action})
        return action_data
    
    def end(self, state_data, reward_data):
        """send the terminal state and the final reward to every agent 
        so they can complete their last transitions

        Args:
            state_data (dict)
            reward_data (dict)
        """
        for n_agent in range(len(state_data)):
                agent_name = state_data[n_agent]["name"]
                agent_state = state_data[n_agent]["state"]
                agent_reward = reward_data[n_agent]["reward"]

                self.agent[agent_name].end(agent_state, agent_reward)
    
    def check(self, action_type:str, action_dim:int, state_dim:int) -> bool:
        assert action_type == agent_action_type[self.type], "env and agent\
             action types don't match"
        is_ok = self.agent.num_actions == action_dim and self.agent.state_dim == state_dim
        return is_ok
    
    def fix(self, action_dim:int, state_dim:int):
        self.agent.adjust_dims(state_dim, action_dim)
