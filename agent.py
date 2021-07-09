from TDAgent import *
from DQN.DQNAgent import *
from GradientPolicyMethods.REINFORCEAgent import *
from GradientPolicyMethods.REINFORCEAgentWithBaseline import *
from GradientPolicyMethods.ActorCriticAgent import *
from GradientPolicyMethods.PPOAgent import *
from GradientPolicyMethods.DDPGAgent import *
from AbaddonAgent import *


class Agent:
    def __init__(
        self, 
        type,
        is_multiagent
    ):
        self.type = type
        self.is_multiagent = is_multiagent

    # ====== Agent execution functions =================================

    def get_agent_action(
        self, state_data, env_type, env_name, reward_data=None, start=False):
        """ Get the agent(s) action in response to the state and reward data.

        Args:
            state_data (dict)
            reward_data (dict, optional): Defaults to None.
            start (bool, optional): indocate if it's the first transition
                                    Defaults to False.

        Returns:
            dict: contains action(s) data
        """
        if self.is_multiagent:
            action_data = self._get_multiagent_action(state_data=state_data,
                                                    reward_data=reward_data,
                                                    start=start)
        else:
            # in case it is a godot env
            if env_type == "godot":
                agent_name = state_data[0]["name"]
                state_data = state_data[0]["state"]
                if not start:
                    reward_data = reward_data[0]['reward']
            # in every case
            action_data = self._get_single_agent_action(agent=self.agent, 
                                        state_data=state_data, 
                                        reward_data=reward_data, 
                                        start=start)
            # if env is Abaddon, format further
            # TODO: about to change
            if env_type == "godot":
                if env_name == "Abaddon-Test-v0":
                    action_data = {
                        "sensor_name": "Radar",
                        "action_name": "dwell",
                        "angle": int(action_data)
                    }
                    action_data = [{"agent_name": agent_name, "action": action_data}]
            
        return action_data
    
    def _get_multiagent_action(self, state_data, reward_data=None, start=False):
        """ distribute states to all agents and get their actions back.

        Args:
            state_data (dict)
            reward_data (dict, optional): Defaults to None.
            start (bool, optional): indicates whether it is the first 
                step of the agent. Defaults to False.

        Returns:
            dict
        """
        action_data = []
        # for each agent, get 
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
      
    def _get_single_agent_action(self, agent, state_data, reward_data=None, start=False):
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
            action_data = agent.start(state_data)
        else:
            action_data = agent.step(state_data, reward_data)
        return action_data

    def end(self, state_data, reward_data):
        if self.is_multiagent:
            self.end_multiagent(state_data, reward_data)
        else:
            self.agent.end(state_data, reward_data)
        
    def end_multiagent(self, state_data, reward_data):
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