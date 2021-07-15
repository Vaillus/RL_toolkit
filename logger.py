import wandb
from typing import Dict, Any, Optional, List


class Logger:
    def __init__(
        self,
        log_every: int = 50,
        wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = {}
    ):
        self.log_counts = {}
        self.log_every = log_every
        self.wandb = wandb
        
        if self.wandb:
            self.init_wandb_project(**wandb_kwargs)
    
    def init_wandb_project(
        self,
        job_type: str, 
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        wandb.init(job_type=job_type, notes=notes, tags=tags, config=config)

    def wandb_watch(self, model, log_freq:int):
        if bool(wandb.run):
            wandb.watch(models=model, log_freq=log_freq)
    
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


