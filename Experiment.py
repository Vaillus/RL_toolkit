from Session import *
import matplotlib.pyplot as plt
from utils import get_params, recursive_get

# === format params functions ==========================================

def format_session_params(session_params_name):
    data = get_params(experiment_params["session_params"])
    session_params = data["session_info"]
    session_params["agent_info"] = data["agent_info"]
    return session_params

def load_experiment_params(experiment_params):
    """
    loads the parameters differently when we want to test the model 
    parameters and when we want to compare different models
    """
    # get the sessions parameters for the same model tested with different parameters
    if experiment_params["experiment_type"] == "parameters testing":
        session_params = format_session_params(experiment_params["session_params"])
        experiment_params["session_info"] = session_params

    # get the sessions parameters for different models
    elif experiment_params["experiment_type"] == "models comparison":
        sessions_params = []
        for session_params_name in experiment_params["session_params"]:
            session_params = format_session_params(session_params_name)
            sessions_params.append(session_params)
        experiment_params["session_info"] = sessions_params

    return experiment_params



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

    # initialization functions =========================================

    def set_params_from_dict(self, params={}):
        self.num_sessions = params.get("num_sessions", 0)
        self.avg_results = params.get("avg_results", False)
        self.avg_length = params.get("avg_length", 100)
        self.experiment_type = params.get("experiment_type", "parameters testing")
        self.environment_name = params.get("environment_name", "unknown environment")

        self.init_sess(params)

        self.environment_name = self.sessions[0].environment_name

    def init_sess(self, params):
        """ Initialize sessions with parameters 
        """
        if self.experiment_type == "parameters testing":
           self.init_sess_param_test(params)
        elif self.experiment_type == "models comparison":
            self.init_sess_model_comp(params)
            
    def init_sess_param_test(self, params):
        # isolate the parameters
        session_params = params.get("session_params")
        session_variants = params.get("session_variants")

        # creating the sessions with their own values
        for n_session in range(self.num_sessions):
            for hyperparam in params["session_variants"].keys():
                session_params = Experiment.modify_session(session_params, 
                                                        session_variants,
                                                        hyperparam,
                                                        n_session)
            self.sessions.append(Session(session_params))

        # storing the hyperparams names for plotting purpose
        for key in params["session_variants"].keys():
            self.varying_params.append((key, params["session_variants"][key]["level"]))
    
    def init_sess_model_comp(self, params):
        sessions_params = params.get("session_info")
        for session_params in sessions_params:
            self.sessions.append(Session(session_params))

    @staticmethod
    def modify_session(session_params, session_variants, hyperparam, n_session):
        level = session_variants[hyperparam]["level"]
        value = session_variants[hyperparam]["values"][n_session]

        if level == "agent":
            session_params["agent_info"][hyperparam] = value
        elif level == "function_approximator":
            session_params["agent_info"]["function_approximator_info"][hyperparam] = value
        elif level == "policy_estimator":
            session_params["agent_info"]["policy_estimator_info"][hyperparam] = value
        
        return session_params
        
    # === main function ================================================

    def run(self):
        rewards_by_session = []
        for session in self.sessions:
            rewards = session.run()
            rewards_by_session.append(rewards)

        rewards_by_session = self.modify_rewards(rewards_by_session)
        self.plot_rewards(rewards_by_session)

    # === plotting functions ===========================================

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
    experiment_params = get_params('experiments/experiment_same_model_params')
    experiment_params = load_experiment_params(experiment_params)
    experiment = Experiment(experiment_params)
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