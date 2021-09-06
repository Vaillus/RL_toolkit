import wandb
from typing import Dict, Any, Optional, List
from functools import reduce  # forward compatibility for Python 3
import operator


class Logger:
    def __init__(
        self,
        log_every: int = 50,
        is_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = {}, 
    ):
        self.log_counts = {}
        self.log_every = log_every
        self.wandb = is_wandb
        if self.wandb:
            wandb.init(**wandb_kwargs)
    
    def init_wandb_project(
        self,
        job_type: str, 
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        wandb.init(job_type, notes=notes, tags=tags, config=config)

    def wandb_watch(self, models, log_freq:Optional[int] = None):
        if log_freq is None:
            log_freq = 1000 # as in the wandb library.
        if bool(wandb.run):
            wandb.watch(models=models, log_freq=log_freq)
    
    def gym_monitor(self):
        # works only when gym generates a video, which it doesn't, for now.
        if bool(wandb.run):
            wandb.gym.monitor()

    def wandb_log(self, log_dict: Dict[str, Any], log_freq = None):
        # log only when a wandb session is launched.
        actual_log_dict = {}
        if bool(wandb.run):
            # increment the values associated with the keys that we want 
            # to log
            for key, value in log_dict.items():
                can_log = self.incr_cnt(key, value, log_freq)
                # if enough data has been accumulated for a key, log it
                # and reset it.
                if can_log:
                    actual_log_dict[key] = self.log_counts[key]["value"]
                    self.log_counts[key]["n_logs"] += 1
                    self.empty_log_count(key)

            wandb.log(actual_log_dict)

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
        #print(wandb.config)
        #print(wandb.config._items)
        kwargs = wandb.config._as_dict()
        del kwargs['_wandb']
        final_kwargs = kwargs.copy()
        
        for key in kwargs.keys():
            keys = key.split('.')
            if len(keys) > 1:
                setInDict(final_kwargs, keys, kwargs[key])
                del final_kwargs[key]
            
        return final_kwargs
    

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

