from Agent import *
from DQN.DQNAgent import *
import gym
import matplotlib.pyplot as plt

def reward_func(env, x, x_dot, theta, theta_dot):  # TODO: do something about it
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


class Session:
    def __init__(self, params={}):
        self.agent = None
        self.environment_name = None
        self.environment = None
        self.num_episodes = None
        self.show = None
        self.show_every = None
        self.plot = None
        self.return_results = None

        self.set_params_from_dict(params=params)

        self.set_env_and_agent(params)

    def set_params_from_dict(self, params={}):
        self.num_episodes = params.get("num_episodes", 100)
        self.show = params.get("show", False)
        self.show_every = params.get("show_every", 10)
        self.environment_name = params.get("environment_name", "MountainCar-v0")
        self.plot = params.get("plot", False)
        self.return_results = params.get("return_results", False)

    def set_env_and_agent(self, params):
        self.environment = gym.make(self.environment_name)
        params["agent_info"]["function_approximator"]["env_min_values"] = self.environment.observation_space.low
        params["agent_info"]["function_approximator"]["env_max_values"] = self.environment.observation_space.high
        self.initialize_agent(params["agent_info"])

    def initialize_agent(self, params={}):
        self.agent = DQNAgent(params)

    def episode(self, episode_id):
        state = self.environment.reset()
        action = self.agent.start(state)
        episode_reward = 0
        done = False
        success = False
        if (self.show is True) and (episode_id % self.show_every == 0):
            print(f'EPISODE: {episode_id}')
        while not done:
            new_state, reward, done, _ = self.environment.step(action)
            x, x_dot, theta, theta_dot = new_state
            reward = reward_func(self.environment, x, x_dot, theta, theta_dot)
            episode_reward += reward

            if (self.show is True) and (episode_id % self.show_every == 0):
                    self.environment.render()


            if not done:
                action = self.agent.step(new_state, reward)
            else:
                self.agent.end(new_state, reward)
                #if new_state[0] >= self.environment.goal_position:
                #    success = True
                return episode_reward, success


    def run(self):
        episode_reward = 0
        success = False
        rewards = np.array([])
        for id_episode in range(self.num_episodes):
            episode_reward, success = self.episode(id_episode)
            rewards = np.append(rewards, episode_reward)
        if self.plot is True:
            plt.plot(rewards)
            plt.show()
            #print(episode_reward)

        if self.return_results:
            return rewards



if __name__ == "__main__":
    session_parameters = {"num_episodes": 300,
                          "plot": True,
                          "show": True,
                          "show_every": 10,
                          "environment_name": "CartPole-v0"}
    agent_parameters = {"num_actions": 2,
                        "is_greedy": False,
                        "epsilon": 0.9,
                        "control_method": "q-learning",
                        "function_approximation_method": "neural network",
                        "function_approximator": {
                            "num_tiles": 4,
                            "num_tilings": 32,
                            "type": "neural network"
                        }}
    function_approx_parameters = {"type": "neural network",
                                  "state_dim": 4,
                                  "action_dim": 2,
                                  "memory_size": 2000,
                                  "update_target_rate": 100,
                                  "batch_size": 128,
                                  "learning_rate": 0.01,
                                  "discount_factor": 0.90
                                }
    agent_parameters["function_approximator"] = function_approx_parameters
    session_parameters["agent_info"] = agent_parameters

    sess = Session(session_parameters)
    sess.run()

