from RL_toolkit.other_agents.perfo_agent import PerfoAgent
import torch
import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt



def test_perfo(state_dim, memory_size, n_hidden_layers, nn_size):
    params = make_perfo_params(state_dim, memory_size, n_hidden_layers, nn_size)
    # init of gpu and cpu agent
    cpu_time = get_agent_exec_stat(params, "cpu")
    gpu_time = get_agent_exec_stat(params, "gpu")
    #print(f" gpu time: {gpu_time} \n cpu time: {cpu_time} \n gain ratio : {cpu_time/gpu_time}")
    return gpu_time, cpu_time

def get_agent_init_stat(params, device_type):
    device = init_device(device_type)
    with cProfile.Profile() as prg:
        agent = PerfoAgent(**params, device=device)
        agent.replay_buffer.noise_init(device=device)
    stats = pstats.Stats(prg)
    return stats.total_tt

def get_agent_exec_stat(params, device_type):
    device = init_device(device_type)
    agent = PerfoAgent(**params, device=device)
    agent.replay_buffer.noise_init(device=device)
    with cProfile.Profile() as prg:
        agent.control()
    stats = pstats.Stats(prg)
    return stats.total_tt

def init_device(device_type):
    if device_type == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_type == "cpu":
        device = torch.device("cpu")
    return device

def make_perfo_params(state_dim, memory_size, n_hidden_layers, nn_size):
    default_params = {
        "state_dim": 100,
        "num_actions": 2,
        "memory_info":{
            "size": 30000},
        "nn_info": {
            "layers_info": {
                "n_hidden_layers": 3,
                "types": "linear",
                "sizes": 200,
                "hidden_activations": "relu",
                "output_activation": "none"
            },
            "optimizer_info": {
                "type": "adam",
                "learning_rate": 0.001
            }
        }
    }
    default_params["state_dim"] = state_dim
    default_params["memory_info"]["size"] = memory_size
    default_params["nn_info"]["layers_info"]["n_hidden_layers"] = n_hidden_layers
    default_params["nn_info"]["layers_info"]["sizes"] = nn_size

    return default_params


if __name__ == "__main__":
    state_dim = 4
    memory_size = 20
    n_hidden_layers = 1
    nn_size = 300
    gpu_times = tuple()
    cpu_times = tuple()
    for i in range(20):
        gpu_time, cpu_time = test_perfo(state_dim, memory_size, n_hidden_layers, nn_size)
        gpu_times = gpu_times + (gpu_time,)
        cpu_times = cpu_times + (cpu_time,)

    plt.plot(np.array(gpu_times))
