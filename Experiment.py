from Session import *
import matplotlib.pyplot as plt
from utils import get_params, set_random_seed, get_from_dict, set_in_dict, get_attr
import random

# === format params functions ==========================================

def load_exp_param(experiment_params):
    """
    loads the parameters differently when we want to test the model 
    parameters and when we want to compare different models
    """
    # get the sessions parameters for the same model tested with 
    # different parameters
    sessions_params = []
    if experiment_params["experiment_type"] == "parameters testing":
        # init sessions params with default session params
        base_sess_params = get_params(experiment_params["session_params_name"])
        sessions_params = [base_sess_params] * experiment_params["num_sessions"]
        # change the params with sessions variants data
        sess_variants = experiment_params["session_variants"]
        sessions_params = modify_sess_params(sessions_params, sess_variants)
    elif experiment_params["experiment_type"] == "models comparison":
        for session_params_name in experiment_params["session_params_names"]:
            session_params = get_params(session_params_name)
            sessions_params.append(session_params)
    experiment_params["sessions_params"] = sessions_params
    return experiment_params

def modify_sess_params(sessions_params, session_variants):
    """iterate through the hyperparams to change and apply them to the 
    session selected

    Args:
        session_params (dict): params of the session before update
        session_variants (dict): all the hyperparams data

    Returns:
        dict: params of the sessions after update
    """
    for hyperparam_data in session_variants:
        sessions_params = change_sess_hyperparam(sessions_params, hyperparam_data)
    return sessions_params

def change_sess_hyperparam(sessions_params, hyperparam_data):
    """Change the hyperparameters in sessions_params for those 
    specified in hyperparam_data

    Args:
        session_params (dict): params of the sessions before update
        hyperparam_data (dict): contain the level, the name and the values
            of the hyperparameter.

    Returns:
        dict: params of the sessions after update
    """
    # separating the hyperparam information
    level = hyperparam_data["level"]
    hp_name = hyperparam_data["param"]
    values = hyperparam_data["values"]
    keys = select_keys(level, hp_name)
    for i in range(len(sessions_params)):
        set_in_dict(sessions_params[i], keys, values[i])
    
    return sessions_params
        
def select_keys(level, hp_name):
    """create the tuple of the keys that give access to the desired 
    hyperparameter

    Args:
        level (str): level in the agent where to find the hyperparam
        hp_name (str): name of the hyperparameter to be accessed

    Returns:
        tuple: keys that lead to the desired hyperparam
    """
    keys = ("agent_info",)
    if (level == "function_approximator") or (level == "policy_estimator"):
        key_name = level + "_info"
        keys += (key_name,)
        if hp_name == "learning_rate":
            keys += ("optimizer_info",)
    keys += (hp_name,)
    return keys

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
        self.varying_params = params.get("session_variants", [])

        self.init_sess(params)
        self.environment_name = self.sessions[0]["environment_name"]

    def init_sess(self, params):
        """Initialize the sessions
        Args:
            params (dict): experiment params
        """
        for sess_data in params["sessions_params"]:
            session_parameters = sess_data["session_info"]
            session_parameters["agent_info"] = sess_data["agent_info"]
            self.sessions.append(session_parameters)
        
    # === main function ================================================

    def run(self):
        """Run the sessions sequentially and store the rewards at the
        end of each session.
        Then plot rewards for the sessions.
        """
        rewards_by_session = []
        for session in self.sessions:
            rewards = session.run()
            rewards_by_session.append(rewards)
        
        rewards_by_session = self.modify_rewards(rewards_by_session)
        self.plot_rewards(rewards_by_session)

    def run_meaningful_session(self):
        # create the list of random seeds
        seeds = []
        rewards_by_session = []
        for i in range(0,5):
            n = random.randint(1,30)
            seeds.append(n)
        seeds = [1]
        for session_param in self.sessions:
            for seed in seeds:
                #set_random_seed(seed)
                session_param["seed"] = seed
                print(session_param)
                #session.set_seed(seed)
                session = Session(session_param)
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

    def generate_legend_text(self, varying_param, id):
        varying_param_name = varying_param["param"]
        param_level = varying_param["level"]
        value = varying_param["values"][id]
        legend = f'{varying_param_name}: {value}'
        return legend

    def plot_rewards(self, rewards_by_session):

        plt.plot(np.array(rewards_by_session).T)
        if self.experiment_type == "parameters testing":
            plt.title(f'Testing {self.sessions[0]["session_type"]} in {self.environment_name}')
        else:
            plt.title(f'Models comparison')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.yscale("linear")
        if self.experiment_type == "parameters testing":
            print(len(self.varying_params[0]["values"]))
            legend = [self.generate_legend_text(self.varying_params[0], id) for id in range(len(self.varying_params[0]["values"]))]
            plt.legend(legend)
        elif self.experiment_type == "models comparison":
            plt.legend([session.session_type for session in self.sessions])
        plt.show()



if __name__ == "__main__":
    exp_param = get_params('experiments/experiment_same_model_params')
    exp_param = load_exp_param(exp_param)
    print(exp_param)
    experiment = Experiment(exp_param)
    experiment.run_meaningful_session()
