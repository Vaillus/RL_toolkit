"""import starship_landing_gym 
import gym

env = gym.make("StarshipLanding-v0", reward_mode="Hugo")
state = env.reset()
print(state)

from stable_baselines3 import ppo
"""

import gym
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 250000000,
    "env_name": "Pendulum-v0",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
#from stable_baselines3.common import make_vec_env
from stable_baselines3.ppo import PPO

# multiprocess environment
#env = make_vec_env('Pendulum-v0', n_envs=4)
env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", gae_lambda=1.0)
model.learn(
    total_timesteps=250000000, 
    callback=WandbCallback()
)