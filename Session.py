import gym
import matplotlib.pyplot as plt

import os

import GodotEnvironment as godot

from typing import Dict, Any, Optional

from agent import AgentInterface
from environment import EnvInterface

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

    def __init__(
        self,
        num_timestep: int,
        plot: Optional[bool] = True,
        show: Optional[bool] = True,
        show_every: Optional[int] = 10,
        return_results: Optional[bool] = True,
        wandb: Optional[bool] = False,
        wandb_name : Optional[str] = "",
        is_multiagent: Optional[bool] = False,
        seed: Optional[int] = 0,
        env_kwargs: Optional[Dict[str, Any]] = {},
        agent_kwargs: Optional[Dict[str, Any]] = {}
    ):
        self.environment = EnvInterface(**env_kwargs)

        self.is_multiagent = is_multiagent
        agent_kwargs["seed"] = seed
        self.agent = AgentInterface(**agent_kwargs)

        self.show = show
        self.show_every = show_every

        self.plot = plot
        self.return_results = return_results

        self.seed = seed
        self.tot_timestep = 0
        self.max_timestep = num_timestep

        self.wandb = wandb
        if self.wandb:
            init_wandb_project(wandb_name)

        self._set_env_and_agent(env_kwargs, agent_kwargs)
        

    # ====== Initialization functions ==================================
    
   
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
            self.environment.set_seed(seed)
            if self.is_multiagent:
                for agent_name in self.agents_names:
                    self.agent[agent_name].set_seed(seed)
            else:
                self.agent.set_seed(seed)



    
    # ==== Main loop functions =========================================
    

    def run(self):
        """Run all the episodes then plot the rewards

        Returns:
            list: the cumulative rewards of the episodes
        """
        episode_reward = 0
        success = False
        rewards = np.array([])
        id_episode = 0
        
        # run the episodes and store the rewards
        while self.tot_timestep < self.max_timestep:
            episode_reward, success, ep_len = self.episode(id_episode)
            self.environment.close()
            if self.show:
                print(f'EPISODE: {id_episode}')
                print(f'reward: {episode_reward}')
                print(f'success: {success}')
            rewards = np.append(rewards, episode_reward)
            wandb.log({
                "General episode info/rewards": episode_reward,
                "General episode info/episode length": ep_len
            })
            id_episode += 1
        
        # plot the rewards
        if self.plot:
            # TODO: change that, it is temporary. We plot the evolution
            # region lighting rate
            if self.environment.type == "Abaddon":
                plt.plot(self.environment.metrics["regions"])
            else:
                avg_reward = Session._average_rewards(rewards)
                avg_reward = np.array(avg_reward)
                plt.plot(avg_reward)
            plt.show()

        if self.environment.type == "probe":
            self.environment.show_result(self.agent)
            
            
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
        state_data = self.environment.reset(episode_id=episode_id)
        action_data = self.agent.get_action(state_data, self.environment.type,
            self.environment.name, start=True)
        # declaration of variables useful in the loop
        episode_reward = 0
        done = False
        success = False
        ep_len = 0
        # Main loop
        while not done:
            ep_len += 1
            self.increment_timestep()
            # run a step in the environment and get the new state, reward 
            # and info about whether the episode is over.
            new_state_data, reward_data, done, _ = self.environment.step(action_data)
            # self.environment.step([float(action)]) | if continuous 
            # mountain car
            reward_data = self.shape_reward(state_data, reward_data)
            # save the reward
            episode_reward = self._save_reward(episode_reward, reward_data)
            # render environment (gym environments only)
            self.environment.render_gym(episode_id)

            if not done:
                # get the action if it's not the last step
                action_data = self.get_agent_action(new_state_data, reward_data)
            else:
                # actions made when the last state is reached
                # get the final reward and success in the mountaincar env
                #reward_data, success = self.assess_mountain_car_success(new_state_data)
                # send parts of the last transition to the agent.
                self.end_agent(new_state_data, reward_data)
                if self.session_type == "REINFORCE" or self.session_type == "REINFORCE with baseline":
                    self.agent.learn_from_experience()
                
                if self.environment_type == "probe":
                    self.environment.plot(episode_id, self.agent)
                
                return episode_reward, success, ep_len



    # === other functions ==============================================



    def print_episode_count(self, episode_id):
        if ((self.show is True) and (episode_id % self.show_every == 0)):
            print(f'EPISODE: {episode_id}')
    
    def shape_reward(self, state_data, reward_data):
        """ shaping reward for cartpole environment
        """
        if self.environment_name == "CartPole-v0" or self.environment_name == "CartPole-v1":  
            x, x_dot, theta, theta_dot = state_data
            reward_data = reward_func(self.environment, x, x_dot, theta, theta_dot)
        if self.environment_name == "MountainCar-v0":
            position = state_data[0]
            if position > 0.5:
                reward_data += 0.1
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
    
    def assess_mountain_car_success(self, new_state_data):
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


    def increment_timestep(self):
        self.tot_timestep += 1
        self.agent.tot_timestep = self.tot_timestep

if __name__ == "__main__":
    # set the working dir to the script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    data = get_params("probe/ddpg_params")
    session_parameters = data["session_info"]
    session_parameters["agent_kwargs"] = data["agent_info"]
    session_parameters["env_kwargs"] = data["environment_info"]

    sess = Session(**session_parameters)
    #sess.set_seed(1)
    #print(sess.agent.policy_estimator.layers[0].weight)
    sess.run()  