import gym
from GodotEnvironment import GodotEnvironment
from probe_env import DiscreteProbeEnv, ContinuousProbeEnv
from typing import Optional, Any


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
        self.env = self._init_env(godot_kwargs, action_type)
        self.show = show
        self.show_every = show_every
        self.seed = 0
        self.action_type = self.get_action_type(action_type)

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
        if self.name.starts_with("Abaddon"):
            return "continuous" # TODO: that will change
        else:
            raise ValueError(f'{self.name} is not supported for action \
                type checking')

    def step(self, action_data):
        self.env.step(action_data)

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
        self.env.show_result(agent)
    
    def render_gym(self, episode_id):
        """ render environment (gym environments only) if specified so
        """
        if (self.show is True) and (episode_id % self.show_every == 0) and (self.type == "gym"):
            self.env.render()
    
    