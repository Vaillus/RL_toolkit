from Session import *
import matplotlib.pyplot as plt
from utils import get_params, recursive_get

# === format params functions ==========================================

# Je vais mettre les paramètres des sessions 

def format_session_params(session_params_name):
    session_params = get_params(session_params_name)
    
    """session_params = data["session_info"]
    session_params["agent_info"] = data["agent_info"]
    """
    return session_params

def load_experiment_params(experiment_params):
    """
    loads the parameters differently when we want to test the model 
    parameters and when we want to compare different models
    """
    # get the sessions parameters for the same model tested with 
    # different parameters
    if experiment_params["experiment_type"] == "parameters testing":
        base_sess_params = format_session_params(experiment_params["session_params_name"])
        
        #experiment_params["session_info"] = session_params
    # get the sessions parameters for different models
    elif experiment_params["experiment_type"] == "models comparison":
        sessions_params = []
        for session_params_name in experiment_params["session_params"]:
            session_params = format_session_params(session_params_name)
            sessions_params.append(session_params)
        #experiment_params["session_info"] = sessions_params

    return experiment_params

def modify_session_params(session_params, session_variants, n_session):
    """iterate through the hyperparams to change and apply them to the 
    session selected

    Args:
        session_params (dict): params of the session before update
        session_variants (dict): all the hyperparams data
        n_session (int): the id of the session that is concerned by the 
            modification.

    Returns:
        dict: params of the session after update
    """
    for hyperparam_data in session_variants:
        session_params = Experiment.modify_session(session_params, 
                                                        hyperparam_data,
                                                        n_session)
    return session_params


class Experiment:
    def __init__(self, params={}):
        self.num_sessions = None
        self.sessions = []
        self.environment_name = None
        self.varying_params = []
        self.avg_results = None
        self.avg_length = None
        self.experiment_type = None
        # test for varying params
        self.base_session_params = None

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
        """Initialize sessions differently if we want to test the same 
        model with varying parameters, or if we want to compare models.
        Args:
            params (dict): experiment params
        """
        # parameters testing case
        if self.experiment_type == "parameters testing":
            # extract params
            session_params_name = params.get("session_params_name")
            session_variants = params.get("session_variants")
            # initialize sessions
            self.init_sess_param_test(session_params_name, session_variants)
            # store varying hyperparams for plotting purposes
            self.store_varying_params(session_variants)

        # models comparison case
        elif self.experiment_type == "models comparison":
            session_params_names = params.get("session_params_names")
            self.init_sess_model_comp(session_params_names)
            
    def init_sess_param_test(self, session_params_name, session_variants):
        """initialize sessions with the same base params, varying those that
        are specified

        Args:
            session_params (dict): base session parameters
            session_variants (dict): varying parameters
        """
        # read the params in the param file with its name
        session_params = get_params(session_params_name)
        # creating the sessions with their own hyperparams
        for n_session in range(self.num_sessions):
            session_params = Experiment.modify_session_params(session_params,
                                                            session_variants,
                                                            n_session)
            # create a session with the params created and add it to the
            # sessions list
            self.sessions.append(Session(session_params))
    
    @staticmethod
    def modify_session_params(session_params, session_variants, n_session):
        """iterate through the hyperparams to change and apply them to the 
        session selected

        Args:
            session_params (dict): params of the session before update
            session_variants (dict): all the hyperparams data
            n_session (int): the id of the session that is concerned by the 
                modification.

        Returns:
            dict: params of the session after update
        """
        for hyperparam_data in session_variants:
            session_params = Experiment.modify_session(session_params, 
                                                            hyperparam_data,
                                                            n_session)
        return session_params

    @staticmethod
    def modify_session(session_params, hyperparam_data, n_session):
        """apply the selected hyperparameter to the selected mission

        Args:
            session_params (dict): params of the session before update
            hyperparam_data (dict): contain the level, the name and the values
                of the hyperparameter.
            n_session (int): the id of the session that is concerned by the 
                modification.

        Returns:
            dict: params of the session after update
        """
        # separating the hyperparam information
        level = hyperparam_data["level"]
        hp_name = hyperparam_data["param"]
        value = hyperparam_data["values"][n_session]
        # adjusting the level in the params where to assign the value to
        # the hyperparam
        if level == "agent":
            session_params["agent_info"][hp_name] = value
        elif level == "function_approximator":
            session_params["agent_info"]["function_approximator_info"][hp_name] = value
        elif level == "policy_estimator":
            session_params["agent_info"]["policy_estimator_info"][hp_name] = value
        
        return session_params

    def store_varying_params(self, session_variants):
        # storing the hyperparams names for plotting purpose
        for session_variant in session_variants:
            self.varying_params.append((session_variant))
    
    def init_sess_model_comp(self, sessions_params_names):
        """If we compare models, just add their session params to sessions.

        Args:
            params (list): the names of the params files to read
        """
        for session_params_name in sessions_params_names:
            session_params = get_params(session_params_name)
            self.sessions.append(Session(session_params))
        
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

    def run_meaningful_session(session):
        pass

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
        varying_param_name = varying_param["param"]
        param_level = varying_param["level"]
        legend = ''
        if param_level == "agent":
            legend = f'{varying_param_name}: {getattr(session.agent,varying_param_name)}'
        elif param_level == "function_approximator":
            legend = f'{varying_param_name}: {getattr(session.agent.function_approximator,varying_param_name)}'
        elif param_level  == "policy_estimator":
            legend = f'{varying_param_name}: {getattr(session.agent.policy_estimator,varying_param_name)}'
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
    exp_param = get_params('experiments/experiment_same_model_params')
    exp_param = load_exp_param(exp_param)
    experiment = Experiment(exp_param)
    experiment.run()
