from TDAgent import *
from DQN.DQNAgent import *
from GradientPolicyMethods.REINFORCEAgent import *
from GradientPolicyMethods.REINFORCEAgentWithBaseline import *
from GradientPolicyMethods.ActorCriticAgent import *
from AbaddonAgent import *
import gym
import matplotlib.pyplot as plt
import json

import os
import pathlib
import sys

import GodotEnvironment as godot

from utils import *


def reward_func(env, x, x_dot, theta, theta_dot):
    """
    For cartpole
    :param env:
    :param x:
    :param x_dot:
    :param theta:
    :param theta_dot:
    :return:
    """
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

class Session:
    def __init__(self, params={}):
        self.agent = None
        self.environment_type = None
        self.environment_name = None
        self.environment = None

        self.num_episodes = None
        self.session_type = None
        self.is_multiagent = None

        self.show = None
        self.show_every = None

        self.plot = None
        self.return_results = None

        self.seed = None

        self.set_params_from_dict(params=params)
        self._set_env_and_agent(params)

    # ====== Initialization functions ==================================

    def set_params_from_dict(self, params={}):
        self.num_episodes = params.get("num_episodes", 100)
        self.show = params.get("show", False)
        self.show_every = params.get("show_every", 10)
        self.environment_type = params.get("environment_type", "gym")
        self.environment_name = params.get("environment_name", "MountainCar-v0")
        self.plot = params.get("plot", False)
        self.return_results = params.get("return_results", False)
        self.session_type = params.get("session_type", "REINFORCE")
        self.is_multiagent = params.get("is_multiagent", False)
        self._init_seed(params.get("seed", None))
 
    def _set_env_and_agent(self, params):
        env_params = params.get("environment_info", {})
        self._init_env(env_params)

        agent_params = params.get("agent_info", {})
        agent_params["seed"] = self.seed
        self._init_agent(agent_params)
    
    def _init_env(self, env_params):
        """the environment is set differently if it's a gym environment 
        or a godot environment.

        Args:
            env_params (dict): used only in case of a godot env.
        """
        if self.environment_type == "gym":
            self.environment = gym.make(self.environment_name)
            self.environment.seed(self.seed)
        elif self.environment_type == "godot":
            self.environment = godot.GodotEnvironment(env_params)
            self.environment.set_seed(self.seed)
    
    def _init_agent(self, agent_params):
        """initialize one or several agents

        Args:
            agent_params (dict)
        """
        if self.is_multiagent:
            # TODO: the following line only works with godot
            self.agents_names = self.environment.agent_names
            self._init_multiagent(agent_params)
        else:
            self.agent = self._init_single_agent(agent_params)

    def _init_multiagent(self, agent_params):
        self.agent = {}
        for agent_name in self.agents_names:
            self.agent[agent_name] = self._init_single_agent(agent_params)


    def _init_single_agent(self, agent_params):
        """Create and return an agent. The type of agent depends on the 
        self.session_type parameter
        Args:
            agent_params (dict)

        Returns:
            Agent: the agent initialized
        """
        agent = None
        if self.session_type == "DQN test":
            agent = DQNAgent(agent_params)
        elif self.session_type == "tile coder test":
            agent = self._init_tc_agent(agent_params)
        elif self.session_type == "REINFORCE":
            agent = REINFORCEAgent(agent_params)
        elif self.session_type == "REINFORCE with baseline":
            agent = REINFORCEAgentWithBaseline(agent_params)
        elif self.session_type == "actor-critic":
            agent = ActorCriticAgent(agent_params)
        elif self.session_type == "Abaddon test":
            agent = AbaddonAgent(agent_params)
        else:
            print("agent not initialized")
        return agent
    
    def _init_tc_agent(self, agent_params):
        """initialization of a tile coder agent, which depends on the 
        gym environment

        Args:
            agent_params (dict)

        Returns:
            Agent
        """
        assert self.environment_name == "gym", "tile coder not supported for godot environments"
        
        params["agent_info"]["function_approximator_info"]["env_min_values"] = \
            self.environment.observation_space.low
        params["agent_info"]["function_approximator_info"]["env_max_values"] = \
            self.environment.observation_space.high
        agent = TDAgent(agent_params)
         
        return agent

    def _init_seed(self, seed):
        self.seed = seed

    def set_seed(self, seed):
        """ Set the Session seed with the param. The set the seed of the
        environment and the agent(s)

        Args:
            seed (int)
        """
        if seed:
            self.seed = seed
            set_random_seed(seed)
            if self.environment_type == "gym":
                self.environment.seed(seed)
            else:
                self.environment.set_seed(seed)
            if self.is_multiagent:
                for agent_name in self.agents_names:
                    self.agent[agent_name].set_seed(seed)
            else:
                self.agent.set_seed(seed)


    # ====== Agent execution functions =================================

    def get_agent_action(self, state_data, reward_data=None, start=False):
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
            if self.environment_type == "godot":
                agent_name = state_data[0]["name"]
                state_data = state_data[0]["state"]
                if not start:
                    reward_data = reward_data[0]['reward']
            # in every case
            action_data = self._get_single_agent_action(agent=self.agent, 
                                        state_data=state_data, 
                                        reward_data=reward_data, 
                                        start=start)
            # if env is abaddon, format further
            # TODO: about to change
            if self.environment_type == "godot":
                if self.environment_name == "Abaddon-Test-v0":
                    action_data = {
                        "sensor_name": "radar",
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

    def end_agent(self, state_data, reward_data):
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

    # ==== Name to be defined ==========================================
    
    def run(self):
        """[summary]

        Returns:
            list: [description]
        """
        episode_reward = 0
        success = False
        rewards = np.array([])
        # run the episodes and store the rewards
        for id_episode in range(self.num_episodes):
            episode_reward, success = self.episode(id_episode)
            self.environment.close()
            if self.show:
                print(f'EPISODE: {id_episode}')
                print(f'reward: {episode_reward}')
                print(f'success: {success}')
            rewards = np.append(rewards, episode_reward)
        # plot the rewards
        if self.plot is True:
            plt.plot(Session._average_rewards(rewards))
            plt.show()
            #print(episode_reward)
        # return the rewards
        
        if self.return_results:
            return rewards

    def episode(self, episode_id):
        """ Run the environment and the agent until the last state is 
        reached.

        Args:
            episode_id (int)

        Returns:
            list, bool: list of the rewards obtained by the agent at 
                        each timestep, a boolean indicating whether the
                        episode was a success.
        """
        # get the first env state and the action that takes the agent
        self.print_episode_count(episode_id=episode_id)
        state_data = self.env_reset(episode_id=episode_id)
        action_data = self.get_agent_action(state_data, start=True)
        # declaration of variables useful in the loop
        episode_reward = 0
        done = False
        success = False

        # Main loop
        while not done:
            # run a step in the environment and get the new state, reward 
            # and info about whether the episode is over.
            new_state_data, reward_data, done, _ = self.environment.step(action_data)
            # self.environment.step([float(action)]) | if continuous 
            # mountain car
            reward_data = self.shape_reward(state_data, reward_data)
            # save the reward
            episode_reward = self._save_reward(episode_reward, reward_data)
            # render environment (gym environments only)
            self.render_gym_env(episode_id)

            if not done:
                # get the action if it's not the last step
                action_data = self.get_agent_action(new_state_data, reward_data)
            else:
                # actions made when the last state is reached
                # get the final reward and success in the mountaincar env
                reward_data, success = self.assess_mc_success(new_state_data)
                # send parts of the last transition to the agent.
                self.end_agent(new_state_data, reward_data)
                if self.session_type == "REINFORCE" or self.session_type == "REINFORCE with baseline":
                    self.agent.learn_from_experience()
                return episode_reward, success

    def env_reset(self, episode_id):
        """ Reset the environment, in both godot and gym case
        """
        if self.environment_type == "godot":
            state_data = self.godot_env_reset(episode_id)
        else:
            state_data = self.environment.reset()
        return state_data

    def godot_env_reset(self, episode_id):
        """ set the right render type for the godot env episode
        """
        render = False
        if (self.show is True) and (episode_id % self.show_every == 0):
            render = True
        state_data = self.environment.reset(render)
        return state_data
    
    def print_episode_count(self, episode_id):
        if ((self.show is True) and (episode_id % self.show_every == 0)):
            print(f'EPISODE: {episode_id}')
    
    def shape_reward(self, state_data, reward_data):
        """ shaping reward for cartpole environment
        """
        if self.environment_name == "CartPole-v0":  
            x, x_dot, theta, theta_dot = state_data
            reward_data = reward_func(self.environment, x, x_dot, theta, theta_dot)

        return reward_data
    
    def _save_reward(self, episode_reward, reward_data):
        """add the reward earned at the last step to the reward 
        accumulated until here

        Args:
            episode_reward (int): reward accumulated until here
            reward_data (int): reward earned at the last step

        Returns:
            int: updated reward
        """
        if self.environment_type == "godot":
            episode_reward += reward_data[0]["reward"]
        else:
            episode_reward += reward_data
        return episode_reward
    
    def render_gym_env(self, episode_id):
        """ render environment (gym environments only) if specified so
        """
        if (self.show is True) and (episode_id % self.show_every == 0) and (self.environment_type != "godot"):
            self.environment.render()
    
    def assess_mc_success(self, new_state_data):
        """ if the environment is mountaincar, assess whether the agent succeeded
        """
        success = False
        reward_data = 0.0
        if self.environment_name == "MountainCar-v0":
            if new_state_data[0] >= self.environment.goal_position:
                success = True
                reward_data = 1

        return reward_data, success 

    @staticmethod
    def _average_rewards(rewards):
        avg_rewards = []
        # transform the rewards to their avergage on the last n episodes 
        # (n being specified in the class parameters)
        for i in range(len(rewards)):  # iterate through rewards
            curr_reward = rewards[i]
            last_n_rewards = [rewards[j] for j in range(i - 100 - 1, i) if j >= 0]
            last_n_rewards.append(curr_reward)
            avg_reward = np.average(last_n_rewards)
            avg_rewards += [avg_reward]

        return avg_rewards

if __name__ == "__main__":
    # set the working dir to the script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data = get_params("abaddon_params")
    session_parameters = data["session_info"]
    session_parameters["agent_info"] = data["agent_info"]
    session_parameters["environment_info"] = data["environment_info"]

    sess = Session(session_parameters)
    #sess.set_seed(1)
    #print(sess.agent.policy_estimator.layers[0].weight)
    sess.run()