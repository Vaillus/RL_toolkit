from TDAgent import *
from DQN.DQNAgent import *
from GradientPolicyMethods.REINFORCEAgent import *
from GradientPolicyMethods.REINFORCEAgentWithBaseline import *
from GradientPolicyMethods.ActorCriticAgent import *
import gym
import matplotlib.pyplot as plt
import json

import os
import pathlib
import sys

#import godot_interface.GodotEnvironment as godot

from utils import *


def reward_func(env, x, x_dot, theta, theta_dot):  # TODO: do something about it
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
        self.show = None
        self.show_every = None
        self.plot = None
        self.return_results = None

        self.session_type = None

        self.is_multiagent = None

        self.set_params_from_dict(params=params)

        self.set_env_and_agent(params)


    # ====== Initialization functions =======================================================


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

    def set_env_and_agent(self, params):
        if self.environment_type == "gym":
            self.environment = gym.make(self.environment_name)
        elif self.environment_type == "godot":
            self.environment = godot.GodotEnvironment(params["environment_info"])
        if self.session_type == "tile coder":  # TODO: might be a problem later
            params["agent_info"]["function_approximator_info"]["env_min_values"] = self.environment.observation_space.low
            params["agent_info"]["function_approximator_info"]["env_max_values"] = self.environment.observation_space.high
        if self.is_multiagent:
            self.agents_names = self.environment.agent_names
            self.initialize_agents(params["agent_info"])
        else:
            self.agent = self.create_agent(params["agent_info"])

    def initialize_agents(self, params={}):
        self.agent = {}
        for agent_name in self.agents_names:
            self.agent[agent_name] = self.create_agent(params)

    def create_agent(self, params={}):
        agent = None
        if self.session_type == "DQN test":
            agent = DQNAgent(params)
        elif self.session_type == "tile coder test":
            agent = TDAgent(params)
        elif self.session_type == "REINFORCE":
            agent = REINFORCEAgent(params)
        elif self.session_type == "REINFORCE with baseline":
            agent = REINFORCEAgentWithBaseline(params)
        elif self.session_type == "actor-critic":
            agent = ActorCriticAgent(params)
        else:
            print("agent not initialized")
        return agent

    # ====== Execution functions =======================================================
    def get_agent_action(self, state_data, reward_data=None, start=False):
        if self.is_multiagent:
            action_data = []
            for n_agent in range(len(state_data)):
                agent_name = state_data[n_agent]["name"]
                agent_state = state_data[n_agent]["state"]
                if start is True:
                    action = self.agent[agent_name].start(agent_state)
                else:
                    agent_reward = reward_data[n_agent]["reward"]
                    action = self.agent[agent_name].step(agent_state, agent_reward)
                action_data.append({"name": agent_name, "action": action})
        else:
            if start is True:
                action_data = self.agent.start(state_data)
            else:
                action_data = self.agent.step(state_data, reward_data)
        return action_data

    def end_agent(self, state_data, reward_data):
        if self.is_multiagent:
            for n_agent in range(len(state_data)):
                agent_name = state_data[n_agent]["name"]
                agent_state = state_data[n_agent]["state"]
                agent_reward = reward_data[n_agent]["reward"]
                self.agent[agent_name].end(agent_state, agent_reward)
        else:
            self.agent.end(state_data, reward_data)

    def episode(self, episode_id):
        # Reset the environment, in both godot and gym case
        if self.environment_type == "godot":
            # set the right render type for the episode
            render = False
            if (self.show is True) and (episode_id % self.show_every == 0):
                render = True
            state_data = self.environment.reset(render)
        else:
            state_data = self.environment.reset()
        
        action_data = self.get_agent_action(state_data, start=True)
        episode_reward = 0
        done = False
        success = False

        if (self.show is True) and (episode_id % self.show_every == 0):
            print(f'EPISODE: {episode_id}')

        # Main loop
        while not done:
            # run a step in the environment and get the new state, reward and info about whether the 
            # episode is over.
            new_state_data, reward_data, done, _ = self.environment.step(action_data) # self.environment.step([float(action)]) | if continuous mountian car

            # shaping reward for cartpole
            if self.environment_name == "CartPole-v0":  # TODO : might want to change that
                x, x_dot, theta, theta_dot = new_state_data
                reward_data = reward_func(self.environment, x, x_dot, theta, theta_dot)
            # save the reward
            episode_reward += reward_data
            # episode_reward += reward_data[0]["reward"] # godot only
            # render environment (gym environments only)
            if (self.show is True) and (episode_id % self.show_every == 0) and (self.environment_type != "godot"):
                    self.environment.render()
            # get the action if it's not the last step
            if not done:
                action_data = self.get_agent_action(new_state_data, reward_data)
            else:
                if self.environment_name == "MountainCar-v0":  # TODO : might want to change that too
                    if new_state_data[0] >= self.environment.goal_position:
                        success = True
                        reward_data = 1
                self.end_agent(new_state_data, reward_data)
                if self.session_type == "REINFORCE" or self.session_type == "REINFORCE with baseline":
                    self.agent.learn_from_experience()
                return episode_reward, success

    def average_rewards(self, rewards):
        avg_rewards = []
        # transform the rewards to their avergage on the last n episodes (n being specified in the class parameters)
        for i in range(len(rewards)):  # iterate through rewards
            curr_reward = rewards[i]
            last_n_rewards = [rewards[j] for j in range(i - 100 - 1, i) if j >= 0]
            last_n_rewards.append(curr_reward)
            avg_reward = np.average(last_n_rewards)
            avg_rewards += [avg_reward]

        return avg_rewards

    def run(self):

        episode_reward = 0
        success = False
        rewards = np.array([])
        # run the episodes and store the rewards
        for id_episode in range(self.num_episodes):
            episode_reward, success = self.episode(id_episode)
            self.environment.close()
            print(f'EPISODE: {id_episode}')
            print(f'reward: {episode_reward}')
            print(f'success: {success}')
            rewards = np.append(rewards, episode_reward)
        # plot the rewards
        if self.plot is True:
            plt.plot(self.average_rewards(rewards))
            plt.show()
            #print(episode_reward)
        # return the rewards
        if self.return_results:
            return rewards



if __name__ == "__main__":
    # '../LonesomeTown/params/first_test_params.json'
    # set the working dir to the script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data = get_params("actor_critic_params")
    session_parameters = data["session_info"]
    session_parameters["agent_info"] = data["agent_info"]

    sess = Session(session_parameters)
    sess.run()
