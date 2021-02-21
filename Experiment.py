from Session import *
import matplotlib.pyplot as plt
from utils import get_params, set_random_seed, get_from_dict, set_in_dict, get_attr
import random
from copy import deepcopy
import math


class Experiment:
    """This class is used for runnning Reinforcement Learning experiments.
    One can use it to compare different models or to compare different sets
    of parameters on a single model. The experiment is done following the
    advices in the paper "Deep Reinforcement Learning That Matters".
    """
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
            rewards = self._smooth_curve(rewards)
            rewards_by_session.append(rewards)
            
        self.plot_rewards(rewards_by_session)

    def run_meaningful_session(self):
        """ Set 5 random seeds, and run each session 5 times with the
        random seeds. 
        The plot the mean of the sessions with std error.
        """
        # create the list of random seeds
        seeds = []
        rewards_by_session = np.array([])
        for i in range(0,5):
            n = random.randint(1,1000)
            seeds.append(n)
        
        # run the sessions n times with the random seeds
        for session_param in self.sessions:
            session_rewards = np.array([])
            for seed in seeds:
                session_param["seed"] = seed
                session = Session(session_param)
                rewards = session.run()
                session_rewards = self._solid_append(session_rewards, rewards)
            rewards_by_session = self._solid_append(rewards_by_session, session_rewards)
        # plot the results
        self.plot_rewards(rewards_by_session)
    
    def _solid_append(self, base_array, added_array):
        if len(base_array) == 0:
            base_array = np.expand_dims(added_array, axis=0)
        else:
            added_array = np.expand_dims(added_array, axis=0)
            base_array = np.concatenate((base_array, added_array), axis=0)
        
        return base_array

    # === plotting functions ===========================================

    def plot_rewards(self, rewards_by_session):
        # generate the curves for each session
        self._generate_plot_curves(rewards_by_session)
        # Generate the text of the plot
        self._generate_plot_labels()
        # show the plot
        plt.show()

    def _generate_plot_curves(self, rewards_by_session):
        """ Plot the mean and std error for each session

        Args:
            rewards_by_session (list): list containing the rewards resulting
            from the several runs for each session (with different seeds)
        """
        for session_reward in rewards_by_session:
            # get the smooth mean and std error curves
            mean_sessions = np.mean(session_reward, axis=0)
            smooth_mean_sessions = self._smooth_curve(mean_sessions)
            std_deviation_sessions = np.std(session_reward, axis=0)
            std_error_sessions = 1.96*(std_deviation_sessions / math.sqrt(len(session_reward)))
            smooth_std_error_sessions = self._smooth_curve(std_error_sessions)
            # plot the std error
            under_line = smooth_mean_sessions - smooth_std_error_sessions
            over_line = smooth_mean_sessions + smooth_std_error_sessions
            x_axis = np.arange(len(smooth_mean_sessions))
            plt.fill_between(x_axis, under_line, over_line, alpha=.1)
            # plot the mean
            plt.plot(smooth_mean_sessions.T, linewidth=2)
    
    def _smooth_curve(self, rewards):
        # smooth the a curve taking the average of the n last samples if
        # required
        if self.avg_results is True:
            avg_rewards = []
            for i in range(len(rewards)):  # iterate through rewards
                # get the previous rewards and the current one
                curr_reward = rewards[i]
                last_n_rewards = [rewards[j] for j in range(i - self.avg_length - 1, i) if j >= 0]
                last_n_rewards.append(curr_reward)
                # average the last n rewards
                avg_reward = np.average(last_n_rewards)
                avg_rewards += [avg_reward]
        return np.array(avg_rewards)

    def _generate_plot_labels(self):
        """ Set the title, the axis labels and the legend of the curves.
        """
        # set the title
        if self.experiment_type == "parameters testing":
            plt.title(f'Testing {self.sessions[0]["session_type"]} in {self.environment_name}')
        else:
            plt.title(f'Models comparison')
        # set axis labels
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.yscale("linear")
        # generate the legends differently if we compare two models or parameters
        if self.experiment_type == "parameters testing":
            legend = [Experiment._generate_legend_text(self.varying_params[0], id) for
                     id in range(len(self.varying_params[0]["values"]))]
        elif self.experiment_type == "models comparison":
            legend = [session["session_type"] for session in self.sessions]
        plt.legend(legend)

    @staticmethod
    def _generate_legend_text(varying_param, id):
        varying_param_name = varying_param["param"]
        param_level = varying_param["level"]
        value = varying_param["values"][id]
        legend = f'{varying_param_name}: {value}'
        return legend

    # === format params functions ======================================

    @staticmethod
    def load_exp_param(experiment_params):
        """
        Load the experiment parameters differently when we want to test variations
        of the model parameters and when we want to compare different models
        """
        sessions_params = []
        # get the sessions parameters for a single model tested with 
        # different parameters
        if experiment_params["experiment_type"] == "parameters testing":
            sessions_params = Experiment._load_exp_params_param_test(experiment_params)
            
        elif experiment_params["experiment_type"] == "models comparison":
            sessions_params = Experiment._load_exp_params_model_comp(experiment_params)

        experiment_params["sessions_params"] = sessions_params
        return experiment_params

    @staticmethod
    def _load_exp_params_param_test(experiment_params):
        """create the sessions params of the variants of a same session from
        the base session params and the parameters variants.

        Args:
            experiment_params (dict)

        Returns:
            list: of session variants params (dict)
        """
        # get the base session params and the variants of the parameters
        base_sess_params = get_params(experiment_params["session_params_name"])
        sess_variants = experiment_params["session_variants"]
        # get a copy of the base session params and change the params with
        # one of the variants. The add it to the list of sessions params.
        sessions_params = []
        for i in range(experiment_params["num_sessions"]):
            cur_sess_params = deepcopy(base_sess_params)
            cur_sess_params = Experiment._modify_sess_params(cur_sess_params, 
                                                            sess_variants, 
                                                            i)
            sessions_params.append(cur_sess_params) 
        return sessions_params

    @staticmethod
    def _load_exp_params_model_comp(experiment_params):
        """ Get the parameters of the models to compare and add them to a 
        list

        Args:
            experiment_params (dict)

        Returns:
            list: of sessions parameters (dict)
        """
        sessions_params = []
        for session_params_name in experiment_params["session_params_names"]:
            session_params = get_params(session_params_name)
            sessions_params.append(session_params)
        
        return sessions_params

    @staticmethod
    def _modify_sess_params(session_params, session_variants, i):
        """ Iterate through the hyperparams to change and apply them to the 
        session selected

        Args:
            session_params (dict): params of the session before update
            session_variants (dict): all the hyperparams data
            i (int): index of session selected

        Returns:
            dict: params of the sessions after update
        """
        for hyperparam_data in session_variants:
            session_params = Experiment._change_sess_hyperparam(session_params,
                                                                hyperparam_data, 
                                                                i)
        return session_params
    
    @staticmethod
    def _change_sess_hyperparam(session_params, hyperparam_data, i):
        """Set the hyperparameters in sessions_params to the value
        specified in hyperparam_data

        Args:
            session_params (dict): params of the sessions before update
            hyperparam_data (dict): contain the level, the name and the values
                of the hyperparameter.
            i (int): index of session selected

        Returns:
            dict: params of the session after update
        """
        # separating the hyperparam information
        level = hyperparam_data["level"]
        hp_name = hyperparam_data["param"]
        value = hyperparam_data["values"][i]
        
        keys = Experiment._select_keys(level, hp_name)
        set_in_dict(session_params, keys, value)
        
        return session_params
    
    @staticmethod
    def _select_keys(level, hp_name):
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


if __name__ == "__main__":
    exp_param = get_params('experiments/experiment_same_model_params')
    exp_param = Experiment.load_exp_param(exp_param)
    #print(exp_param)
    experiment = Experiment(exp_param)
    experiment.run_meaningful_session()
