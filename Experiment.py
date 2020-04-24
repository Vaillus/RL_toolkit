from Session import *
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, params={}):
        self.num_sessions = None
        self.sessions = []
        self.environment_name = None
        self.varying_params = []
        self.avg_results = None
        self.avg_length = None

        self.set_params_from_dict(params)

    def set_params_from_dict(self, params={}):
        self.num_sessions = params.get("num_sessions", 0)
        self.avg_results = params.get("avg_results", False)
        self.avg_length = params.get("avg_length", 10)

        self.init_sessions(params)

    def init_sessions(self, params):
        session_params = params.get("session_info")
        agent_params = session_params.get("agent_info")
        # creating the sessions with their own values
        for n_session in range(self.num_sessions):
            for key in params["session_variants"].keys():
                agent_params[key] = params["session_variants"][key][n_session]
                session_params["agent_info"] = agent_params
            self.sessions.append(Session(session_params))

        # TODO: find another system for this
        for key in params["session_variants"].keys():
            self.varying_params.append(key)


        print(self.sessions)

    def run(self):
        rewards_by_session = []
        for session in self.sessions:
            rewards = session.run()
            rewards_by_session.append(rewards)

        rewards_by_session = self.modify_rewards(rewards_by_session)
        self.plot_rewards(rewards_by_session)

    def modify_rewards(self, rewards_by_session):
        rewards_to_return = rewards_by_session
        # transform the rewards to their avergage on the last n episodes (n being specified in the class parameters)
        if self.avg_results is True:
            avg_rewards_by_session = []

            for rewards in rewards_by_session:  # split the rewards sequences by episode
                avg_rewards = []
                for i in range(len(rewards)):  # we iterate through rewards
                    curr_reward = rewards[i]
                    last_n_rewards = [rewards[j] for j in range(i - self.avg_length - 1, i) if j >= 0]
                    last_n_rewards.append(curr_reward)
                    avg_reward = np.average(last_n_rewards)
                    avg_rewards += [avg_reward]
                avg_rewards_by_session.append(avg_rewards)
            rewards_to_return = avg_rewards_by_session

        return rewards_to_return

    def plot_rewards(self, rewards_by_session):


        plt.plot(np.array(rewards_by_session).T)
        plt.xlabel("Episode")
        plt.ylabel("reward Per Episode")
        plt.yscale("linear")
        plt.legend(
            [[f'{varying_param}: {getattr(session.agent,varying_param)}' for varying_param in self.varying_params] for
             session in self.sessions])
        plt.show()



if __name__ == "__main__":
    agent_parameters = {"num_actions": 3,
                        "is_greedy": False,
                        "epsilon": 0.999,
                        "learning_rate": 0.4,
                        "discount_factor": 1,
                        "control_method": "expected sarsa",
                        "function_approximation_method": "tile coding",
                        "function_approximator": {
                            "num_tiles": 4,
                            "num_tilings": 32
                            }
                        }


    session_parameters = {"num_episodes": 100,
                          "return_results": True}

    experiment_parameters = {"num_sessions":  2,
                             "session_variants": {
                                "control_method": ["expected sarsa", "q-learning"]
                                },
                             "avg_results":True
                             }


    session_parameters["agent_info"] = agent_parameters
    experiment_parameters["session_info"] = session_parameters

    experiment = Experiment(experiment_parameters)
    experiment.run()

