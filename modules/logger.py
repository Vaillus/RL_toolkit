import wandb
from typing import Dict, Any, Optional, List
from functools import reduce  # forward compatibility for Python 3
import operator
import re
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Logger:
    def __init__(
        self,
        log_every: int = 50,
        is_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = {}, 
        video_record: Optional[bool] = False,
        record_every: Optional[int] = 100,
        ep_freq: Optional[int] = 300,
        agent_freq: Optional[int] = 300,
        grad_freq: Optional[int] = 300,
        is_print: Optional[bool] = False
    ):
        self.log_counts = {}
        self.log_every = log_every
        self.wandb = is_wandb
        if self.wandb:
            wandb.init(**wandb_kwargs, monitor_gym=True)
        self.is_print = is_print
        self.video_record = video_record
        self.record_every = record_every
        self.rec = None
        self.n_ep = 0
        self.ep_freq = ep_freq
        self.agent_freq = agent_freq
        self.grad_freq = grad_freq
    
    def init_wandb_project(
        self,
        job_type: str, 
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        wandb.init(job_type, notes=notes, tags=tags, config=config, monitor_gym=True)

    def wandb_watch(self, models, log_freq:Optional[int] = None, type = None):
        if type is not None:
            attr_str = type + "_freq"
            log_freq = getattr(self, attr_str)
        if log_freq is None:
            log_freq = 1000  # as in the wandb library.
        if bool(wandb.run):
            wandb.watch(models=models, log_freq=log_freq)
    
    def gym_init_recording(self, env):
        """ Called once after the gym environment has been created. 
        Associates a recorder to the environment.
        """
        # plotting reward and stuff.
        if bool(wandb.run) and self.video_record:
            #self.rec = VideoRecorder(env, base_path="./video/test")
            #wandb.gym.monitor()
            pass
        
    """def gym_capture_frame(self, n_ep=None):
        if n_ep is not None:
            print(n_ep)
            self.n_ep = n_ep
        if self.n_ep % self.record_every == 0:
            print("ye")
            self.rec.capture_frame()"""

    def log(self, log_dict: Dict[str, Any], log_freq = None, type = None):
        actual_log_dict = {}
        assert (log_freq is None) != (type is None)
        # increment the values associated with the keys that we want 
        # to log
        if type is not None:
            attr_str = type + "_freq"
            log_freq = getattr(self, attr_str)
        for key, value in log_dict.items():
            can_log = self.incr_cnt(key, value, log_freq)
            # if enough data has been accumulated for a key, log it
            # and reset it.
            if can_log:
                actual_log_dict[key] = self.log_counts[key]["value"]
                self.log_counts[key]["n_logs"] += 1
                self.empty_log_count(key)
        if self.wandb:
            self.wandb_log(actual_log_dict)
        if type == "ep":
            self.regular_print(actual_log_dict)

    def regular_print(self, log_dict: Dict[str, Any]):
        # if the log dict is not empty
        if bool(log_dict):
            print("EPISODE: ", self.n_ep)
            for key, value in log_dict.items():
                key = key.split("/")[-1]
                print(f"{key}: {value}")

    def wandb_log(self, log_dict: Dict[str, Any]):
        # log only when a wandb session is launched.
        if bool(wandb.run):
            wandb.log(log_dict)

    def wandb_plot(self, plot_dict):
        if bool(wandb.run):
            wandb.log(plot_dict)
    
    def incr_cnt(self, key: str, value: float, log_freq=None):
        if not key in self.log_counts:
            self.log_counts[key] = self.create_log_cnt(log_freq)
        self.log_counts[key]["count"] += 1
        self.log_counts[key]["value"] += value / self.log_counts[key]["log_freq"]
        
        if self.log_counts[key]["count"] == self.log_counts[key]["log_freq"]:
            return True
        else:
            return False
    
    def create_log_cnt(self, log_freq=None):
        if log_freq is None:
            log_freq = self.log_every
        log_cnt = {
            "count": 0,
            "value": 0,
            "n_logs": 0,
            "log_freq": log_freq
        }
        
        return log_cnt
    
    def empty_log_count(self, key):
        self.log_counts[key]["count"] = 0
        self.log_counts[key]["value"] = 0

    def get_config(self):
        """ I am having trouble with the wandb parameters that fail to load correctly.
        In order to correct this, I made this function which interprets the parameters
        correctly. For example, 
        policy_estimator_info.layers_info.sizes.1
        will be interpreted as:
        ["policy_estimator_info"]["layers_info"]["sizes"][1]

        Returns:
            [type]: [description]
        """
        kwargs = wandb.config._as_dict()
        del kwargs['_wandb']
        final_kwargs = kwargs.copy()
        for key in kwargs.keys():
            keys = key.split('.')
            if len(keys) > 1:
                if keys[-1].isdigit():
                    id = int(keys.pop(-1))
                    #print(f"kwargs:{kwargs},\n  keys: {keys}")
                    # TODO: crashes with sweeps, when we try to access an id 
                    # of a neural network that wasn't present in the original
                    # parameters (e.g.: function_approximator_info.layers_info.sizes.0)
                    test_list = getFromDict(kwargs, keys)
                    test_list[id] = kwargs[key]
                    new_value = test_list
                else: 
                    new_value = kwargs[key]
                setInDict(final_kwargs, keys, new_value)
                del final_kwargs[key]
            
        return final_kwargs
    

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
