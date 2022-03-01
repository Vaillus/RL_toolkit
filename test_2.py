import starship_landing_gym 
import gym

env = gym.make("StarshipLanding-v0", reward_mode="Hugo")
state = env.reset()
print(state)