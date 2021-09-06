import gym
from GodotEnvironment import GodotEnvironment
from modules.probe_env import DiscreteProbeEnv, ContinuousProbeEnv
from typing import Dict, Optional, Any
import re
from modules.logger import Logger
from utils import get_params

class EnvInterface:
    def __init__(
        self,
        type: str,
        name: str,
        godot_kwargs: Optional[dict] = {},
        show: Optional[bool] = True,
        show_every: Optional[int] = 10,
        action_type: Optional[str] = "discrete"
    ):
        self.type = type
        self.name = name
        
        self.show = show
        self.show_every = show_every
        self.seed = 0
        self.env = self._init_env(godot_kwargs, action_type)
        self.action_type = self.get_action_type(action_type)
        self.logger = None

    def _init_env(self, godot_kwargs, action_type):
        if self.type == "gym":
            env = gym.make(self.name)
            env.seed(self.seed)
        elif self.type == "godot":
            env = GodotEnvironment(godot_kwargs)
            env.set_seed(self.seed)
        elif self.type == "probe":
            if action_type == "discrete":
                env = DiscreteProbeEnv(self.name)
            elif action_type == "continuous":
                env = ContinuousProbeEnv(self.name)
        return env
    
    def set_seed(self, seed:int):
        if self.type == "gym":
            self.env.seed(seed)
        else:
            self.env.set_seed(seed)
    
    def set_logger(self, logger:Logger):
        self.logger = logger
        if self.type == "probe":
            self.env.set_logger(logger)
        elif self.type == "gym":
            self.logger.gym_monitor()

    def get_action_type(self, action_type:str) -> str:
        if self.type == "probe":
            return action_type
        elif self.type == "gym":
            return self.get_gym_action_type()
        elif self.type == "godot":
            return self.get_godot_action_type()
    
    def get_gym_action_type(self):
        if self.name.startswith("MountainCarContinous"):
            return "continuous"
        elif self.name.startswith("MountainCar"):
            return "discrete"
        elif self.name.startswith("CartPole"):
            return "discrete"
        elif self.name.startswith("Pendulum"):
            return "continuous"
        elif self.name.startswith("Acrobot"):
            return "discrete"
        else:
            raise ValueError(f'{self.name} is not supported for action \
                type checking')
    
    def get_godot_action_type(self):
        if self.name.startswith("Abaddon"):
            return "continuous" # TODO: that will change
        else:
            raise ValueError(f'{self.name} is not supported for action \
                type checking')

    def step(self, action_data):
        if self.name.startswith("Pendulum"):
            action_data *= 2.0
        return self.env.step(action_data) # type problem somewhere # .detach().numpy()

    def close(self):
        self.env.close()
    
    def reset(self, episode_id):
        """ Reset the environment, in both godot and gym case
        """
        if self.type == "godot":
            state_data = self.godot_env_reset(episode_id)
        else:
            state_data = self.env.reset()
        return state_data

    def godot_env_reset(self, episode_id):
        """ set the right render type for the godot env episode
        """
        render = False
        if (self.show is True) and (episode_id % self.show_every == 0):
            render = True
        state_data = self.env.reset(render)
        return state_data

    def show_result(self, agent):
        self.env.show_results(agent)
    
    def render_gym(self, episode_id):
        """ render environment (gym environments only) if specified so
        """
        if (self.show is True) and (episode_id % self.show_every == 0) and (self.type == "gym"):
            self.env.render()
    
    def unwrap_godot_state_data(self, state_data, reward_data, start):
        agent_name = ""
        if self.type == "godot":
            agent_name = state_data[0]["name"]
            state_data = state_data[0]["state"]
            if not start:
                reward_data = reward_data[0]['reward']
        
        return state_data, reward_data, agent_name
    
    def wrap_godot_action_data(self, action_data:int, agent_name:str) -> dict:
        if self.type == "godot":
            if self.name.startswith("Abaddon"):
                action_data = {
                    "sensor_name": "Radar",
                    "action_name": "dwell",
                    "angle": int(action_data)
                }
                action_data = [{"agent_name": agent_name, "action": action_data}]
        return action_data
    
    def assess_mountain_car_success(self, new_state_data):
        """ if the environment is mountaincar, assess whether the agent succeeded
        """
        success = False
        reward_data = 0.0
        if self.name.startswith("MountainCar"):
            if new_state_data[0] >= self.env.goal_position:
                success = True
                reward_data = 1.0

        return reward_data, success 
    
    def shape_reward(self, state_data, reward_data):
        """ shaping reward for cartpole environment
        """
        if self.name.startswith("CartPole"):  
            x, x_dot, theta, theta_dot = state_data
            reward_data = self.reward_func(x, x_dot, theta, theta_dot)
        if self.name.startswith("MountainCar"):
            position = state_data[0]
            if position > 0.5: # why?
                reward_data += 0.1
        return reward_data
    
    def reward_func(self, x, x_dot, theta, theta_dot):
        """ For cartpole only
        """
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.5
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward
    
    def plot(self, episode_id, agent):
        if self.type == "probe":
            self.env.plot(episode_id, agent)
        else:
            raise ValueError("cannot use plot with something else than a \
                probe environment")
    
    def get_env_data(self) -> Dict[str, Any]:
        """ Get the env data useful for agent correct initialization
        
        Raises:
            ValueError: [description]

        Returns:
            Dict[str, Any]: keys: [action_type, action_dim, state_dim]
        """
        dict_envs = get_params("misc/env_data")
        if self.type == "gym":
            env_data = dict_envs[self.type][re.split("-", self.name)[0]]
        elif self.type == "probe":
            env_data = dict_envs[self.type][self.action_type][self.name]
        else:
            raise ValueError(f"{self.type} not supported for env-agent matching")
        return env_data

