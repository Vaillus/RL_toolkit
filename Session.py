import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from modules.agent import MultiAgentInterface, SingleAgentInterface
from modules.environment import EnvInterface
from modules.logger import Logger

from utils import get_params, set_random_seed
import wandb

def get_video_file():
    files = os.listdir("./video/")
    for file in files:
        if file.endswith(".mp4"):
            return file
    return None

class Session:
    def __init__(
        self,
        num_timestep: int,
        plot: Optional[bool] = True,
        show: Optional[bool] = True,
        show_every: Optional[int] = 10,
        return_results: Optional[bool] = True,
        is_multiagent: Optional[bool] = False,
        seed: Optional[int] = 0,
        env_kwargs: Optional[Dict[str, Any]] = {},
        agent_kwargs: Optional[Dict[str, Any]] = {},
        logger_kwargs : Optional[Dict[str, Any]] = {}
    ):
        self.environment = EnvInterface(
            **env_kwargs, 
            show=show, 
            show_every=show_every)
        
        if logger_kwargs and logger_kwargs.get('is_wandb', False):
            logger_kwargs['wandb_kwargs']['config'] = agent_kwargs["agent_kwargs"]
        self.logger = Logger(**logger_kwargs)

        self.is_multiagent = is_multiagent
        agent_kwargs["seed"] = seed # those two might not work for multiagent.
        self.agent = self._init_agent(agent_kwargs, is_multiagent)

        self.show = show
        self.show_every = show_every

        self.plot = plot
        self.return_results = return_results

        self.seed = seed
        self.tot_timestep = 0
        self.max_timestep = num_timestep

        
        self.agent.set_logger(self.logger)
        self.environment.set_logger(self.logger)

        self.adjust_agent_with_env()
        

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
            self.agent.set_seed(seed)
    
    def _init_agent(self, agent_kwargs, is_multiagent):
        if is_multiagent:
            return MultiAgentInterface(**agent_kwargs, logger=self.logger)
        else:
            return SingleAgentInterface(**agent_kwargs, logger=self.logger)

    
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
                pass
                #print(f'EPISODE: {id_episode}')
                #print(f'reward: {episode_reward}')
                #print(f'success: {success}')
            rewards = np.append(rewards, episode_reward)
            self.logger.wandb_log({
                "rewards": episode_reward,
                "General/episode length": ep_len
            }, type="ep")
            #wandb.log({"gameplays": wandb.Video("./video/"+get_video_file(), caption='episode: '+str(id_episode), fps=4, format="mp4"), "step": id_episode})
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
        action_data = self.get_agent_action(state_data, start=True)
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
            reward_data = self.environment.shape_reward(state_data, reward_data)
            # save the reward
            episode_reward = self._save_reward(episode_reward, reward_data)
            # render environment (gym environments only)
            self.environment.render(episode_id)
            # TODO: register
            if not done:
                # get the action if it's not the last step
                action_data = self.get_agent_action(new_state_data, reward_data)
            else:
                # actions made when the last state is reached
                # get the final reward and success in the mountaincar env
                #reward_data, success = self.environment.assess_mountain_car_success(new_state_data)
                # send parts of the last transition to the agent.
                self.agent.end(new_state_data, reward_data)
                if self.agent.type.startswith("REINFORCE"):
                    self.agent.learn_from_experience()
                
                if self.environment.type == "probe":
                    self.environment.plot(episode_id, self.agent.agent)
                
                return episode_reward, success, ep_len


    # === other functions ==============================================

    def get_agent_action(self, state_data, reward_data= {}, start=False):
        # TODO : my intuition is that the two environment functions work
        # only with single agent and not multiagent.
        # in case of an Abaddon, unwrap the agent name, the state and 
        # the reward data from the state data
        state_data, reward_data, agent_name = \
            self.environment.unwrap_godot_state_data(
                state_data, reward_data, start)

        action_data = self.agent.get_action(state_data, reward_data, start, 
                                            self.tot_timestep)

        # in the case of an Abaddon environment, wrap the state data in a certain way.
        action_data = self.environment.wrap_godot_action_data(action_data, agent_name)
        if type(action_data) == torch.Tensor: # TODO: why?
            action_data = action_data.detach().numpy()

        return action_data

    def print_episode_count(self, episode_id):
        if ((self.show is True) and (episode_id % self.show_every == 0)):
            pass
            #print(f'EPISODE: {episode_id}')
    
    def _save_reward(self, episode_reward, reward_data):
        """add the reward earned at the last step to the reward 
        accumulated until here

        Args:
            episode_reward (int): reward accumulated until here
            reward_data (int): reward earned at the last step

        Returns:
            int: updated reward
        """
        if self.environment.type == "godot":
            episode_reward += reward_data[0]["reward"]
        else:
            episode_reward += reward_data
        return episode_reward

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


    # === agent-env wrapping ===========================================


    def adjust_agent_with_env(self):
        env_data = self.environment.get_env_data()
        agent_ok = self.agent.check(**env_data)
        if not agent_ok:
            self.agent.fix(env_data["action_dim"], env_data["state_dim"])
        

if __name__ == "__main__":
    # set the working dir to the script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    data = get_params("other/ppo_params")
    session_parameters = data["session_info"]
    session_parameters["agent_kwargs"] = data["agent_info"]
    session_parameters["env_kwargs"] = data["env_info"]
    session_parameters["logger_kwargs"] = data["logger_kwargs"]

    sess = Session(**session_parameters)
    #sess.set_seed(1)
    #print(sess.agent.policy_estimator.layers[0].weight)
    sess.run()
    print("done")
