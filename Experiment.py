from Session import *
import matplotlib.pyplot as plt

def load_experiment_params(path):
    """
    loads the parameters differently when we want to test the model parameters and when we want to compare
    different models
    """
    # get the experiment parameters
    with open(path) as json_file:
        experiment_parameters = json.load(json_file)

    # get the sessions parameters for the same model tested with different parameters
    if experiment_parameters["experiment_type"] == "parameters testing":
        with open(experiment_parameters["path_to_model_params"]) as json_file:
            data = json.load(json_file)
            session_parameters = data["session_info"]
            session_parameters["agent_info"] = data["agent_info"]
        experiment_parameters["session_info"] = session_parameters

    # get the sessions parameters for different models
    elif experiment_parameters["experiment_type"] == "models comparison":
        sessions_parameters = []
        for model_path in experiment_parameters["path_to_model_params"]:
            with open(model_path) as json_file:
                data = json.load(json_file)
                session_parameters = data["session_info"]
                session_parameters["agent_info"] = data["agent_info"]
            sessions_parameters.append(session_parameters)
            experiment_parameters["session_info"] = sessions_parameters

    return experiment_parameters


class Experiment:
    def __init__(self, params={}):
        self.num_sessions = None
        self.sessions = []
        self.environment_name = None
        self.varying_params = []
        self.avg_results = None
        self.avg_length = None
        self.experiment_type = None

        self.set_params_from_dict(params)

    # initialization functions ============================================================================

    def set_params_from_dict(self, params={}):
        self.num_sessions = params.get("num_sessions", 0)
        self.avg_results = params.get("avg_results", False)
        self.avg_length = params.get("avg_length", 100)
        self.experiment_type = params.get("experiment_type", "parameters testing")

        self.init_sessions(params)

    def init_sessions(self, params):
        """ Initialize sessions with parameters """
        if self.experiment_type == "parameters testing":
            # isolate the parameters
            session_params = params.get("session_info")
            agent_params = session_params.get("agent_info")
            function_approximator_params = agent_params.get("function_approximator_info")
            policy_estimator_params = agent_params.get("policy_estimator_info")

            # creating the sessions with their own values
            for n_session in range(self.num_sessions):
                for key in params["session_variants"].keys():
                    if params["session_variants"][key]["level"] == "agent":
                        agent_params[key] = params["session_variants"][key]["values"][n_session]
                    elif params["session_variants"][key]["level"] == "function_approximator":
                        function_approximator_params[key] = params["session_variants"][key]["values"][n_session]
                    elif params["session_variants"][key]["level"] == "policy_estimator":
                        policy_estimator_params[key] = params["session_variants"][key]["values"][n_session]

                    agent_params["function_approximator_info"] = function_approximator_params
                    agent_params["policy_estimator_info"] = policy_estimator_params
                    session_params["agent_info"] = agent_params
                self.sessions.append(Session(session_params))

            for key in params["session_variants"].keys():
                self.varying_params.append((key, params["session_variants"][key]["level"]))

        elif self.experiment_type == "models comparison":
            sessions_params = params.get("session_info")
            for session_params in sessions_params:
                self.sessions.append(Session(session_params))

    # main function ===================================================================================================

    def run(self):
        rewards_by_session = []
        for session in self.sessions:
            rewards = session.run()
            rewards_by_session.append(rewards)

        rewards_by_session = self.modify_rewards(rewards_by_session)
        self.plot_rewards(rewards_by_session)

    # plotting functions ==============================================================================================

    def modify_rewards(self, rewards_by_session):
        rewards_to_return = rewards_by_session
        # transform the rewards to their avergage on the last n episodes (n being specified in the class parameters)
        if self.avg_results is True:
            avg_rewards_by_session = []

            for rewards in rewards_by_session:  # split the rewards sequences by episode
                avg_rewards = []
                for i in range(len(rewards)):  # iterate through rewards
                    curr_reward = rewards[i]
                    last_n_rewards = [rewards[j] for j in range(i - self.avg_length - 1, i) if j >= 0]
                    last_n_rewards.append(curr_reward)
                    avg_reward = np.average(last_n_rewards)
                    avg_rewards += [avg_reward]
                avg_rewards_by_session.append(avg_rewards)
            rewards_to_return = avg_rewards_by_session

        return rewards_to_return

    def generate_legend_text(self, varying_param, session):
        legend = ''
        if varying_param[1] == "agent":
            legend = f'{varying_param[0]}: {getattr(session.agent,varying_param[0])}'
        elif varying_param[1] == "function_approximator":
            legend = f'{varying_param[0]}: {getattr(session.agent.function_approximator,varying_param[0])}'
        elif varying_param[1]  == "policy_estimator":
            legend = f'{varying_param[0]}: {getattr(session.agent.policy_estimator,varying_param[0])}'
        return legend

    def plot_rewards(self, rewards_by_session):

        plt.plot(np.array(rewards_by_session).T)
        plt.title(f'Models comparison in {self.environment_name}')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.yscale("linear")
        if self.experiment_type == "parameters testing":
            plt.legend([[self.generate_legend_text(varying_param, session) for varying_param in self.varying_params] for
                    session in self.sessions])
        elif self.experiment_type == "models comparison":
            plt.legend([session.session_type for session in self.sessions])
        plt.show()



if __name__ == "__main__":
    experiment_path = 'params/experiment_different_models_params.json'
    experiment_parameters = load_experiment_params(experiment_path)
    experiment = Experiment(experiment_parameters)
    experiment.run()

"""
 session_parameters = {"num_episodes": 500,
                       "environment_name": "CartPole-v0",
                       "return_results": True}

 agent_parameters = {"num_actions": 2,
                     "is_greedy": False,
                     "epsilon": 0.95,
                     "control_method": "q-learning",
                     "function_approximation_method": "tile coder",
                     "discount_factor": 1,
                     "learning_rate": 0.1,
                     "function_approximator_info": {
                         "num_tiles": 4,
                         "num_tilings": 32,
                         "type": "tile coder"
                     }}

 # tile coder
 "function_approximator_info": {
                         "num_tiles": 4,
                         "num_tilings": 32,
                         "type": "tile coder"
                     }
 # neural network
 "function_approximator_info": {
                         "type": "neural network",
                         "state_dim": 4,
                         "action_dim": 2,
                         "memory_size": 1000,
                         "update_target_rate": 100,
                         "batch_size": 128,
                         "learning_rate": 0.01,
                         "discount_factor": 0.90
                     }}                        

 agent_parameters = {"num_actions": 3,
                     "is_greedy": False,
                     "epsilon": 0.9,
                     }
 function_approx_parameters = {"type": "neural network",
                               "state_dim": 4,
                               "action_dim": 2,
                               "memory_size": 1000,
                               "update_target_rate": 100,
                               "batch_size": 128,
                               "learning_rate": 0.01,
                               "discount_factor": 0.90
                               }

 agent_parameters["function_approximator_info"] = function_approx_parameters
 """