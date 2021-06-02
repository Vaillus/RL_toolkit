import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import set_random_seed

from torch.utils.tensorboard import SummaryWriter
#import torchvision


class CustomNeuralNetwork(nn.Module):
    """ Attempt to make an easy-to-use neural net class
    """
    def __init__(self, params):
        super(CustomNeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = []
        self.optimizer = None
        self.seed = None
        self.history = np.array([])

        self.set_params_from_dict(params=params)

    
    def set_params_from_dict(self, params):
        self.init_layers(params["layers_info"])
        self.init_optimizer(params["optimizer_info"])
        self.seed = params.get("seed", None)
        if self.seed:
            torch.manual_seed(self.seed)
    
    def set_seed(self, seed):
        self.seed = seed
        set_random_seed(seed)
        for i in range(len(self.layers)):
            self.layers[i].weight.data.normal_(0, 0.1)


    def init_layers(self, layers_info):
        for layer_info in layers_info:
            if layer_info["type"] == "linear":
                layer = nn.Linear(layer_info["input_size"], layer_info["output_size"])
                layer.weight.data.normal_(0, 0.1)  
            
            self.layers.append(layer)
            # next line could work. It is useful mainly on a policy network, apparently.
            #self.layers[-1].weight.data *= 0.01
            self.activations.append(layer_info["activation"])

    def init_optimizer(self, optimizer_info):
        if optimizer_info["type"] == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=optimizer_info["learning_rate"])

    def forward(self, x):
        # format the input data
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, list):
            x = torch.Tensor(x)
        elif isinstance(x, tuple):
            x = torch.Tensor(x)
        x = x.float()
        num_layers = len(self.layers)
        for i in range(num_layers):
            if self.activations[i] == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activations[i] == "tanh":
                x = F.tanh(self.layers[i](x))
            elif self.activations[i] == "softmax":
                x = F.softmax(self.layers[i](x))
            elif self.activations[i] == "none":
                x = self.layers[i](x)
            else:
                x = self.layers[i](x)

        return x

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()  # il faut que la loss ait une seule valeur.
        self.optimizer.step()

    # === functions related to gradient logging and plotting ===========
    # === === logging ==================================================
    def add_state_to_history(self):
        z_value = 1.96
        nn_state = np.array([])
        for layer in self.layers:
            layer_state = self.create_layer_state(layer)
            nn_state = self.add_lay2nn_state(nn_state, layer_state)
        self.add_state2history(nn_state)
        

    def create_layer_state(self, layer):
        z_value = 1.96
        weight = layer.weight.data
        grad = layer.weight.grad
        layer_size = self.get_layer_size(layer)
        weight_std_error = z_value * weight.std() / math.sqrt(layer_size)
        grad_std_error = z_value * grad.std() / math.sqrt(layer_size)
        layer_state = [ (weight.mean(), weight_std_error),
                        (grad.mean(), grad_std_error)
                        ]
        return layer_state

    def add_lay2nn_state(self,  nn_state: np.ndarray, 
                                layer_state: np.ndarray) -> np.ndarray:
        """add the layer state to the neural net state 
        """
        layer_state = np.expand_dims(layer_state, 0)
        if nn_state.size == 0:
            nn_state = layer_state
        else:
            nn_state = np.append(nn_state, layer_state, axis=0)
        return nn_state
    
    def add_state2history(self, nn_state):
        nn_state = np.expand_dims(nn_state, 0)
        if self.history.size == 0:
            self.history = nn_state
        else:
            self.history = np.append(self.history, nn_state, axis=0)

    def get_layer_size(self, layer):
        size = layer.in_features * layer.out_features
        return size

    # === === plotting =================================================
    
    def plot_weights(self, layer_n: int):
        mean = self.history[:, layer_n, 0, 0]
        std_err = self.history[:, layer_n, 0, 1]
        smooth_mean = self._smooth_curve(mean)
        under_line = smooth_mean - std_err
        over_line = smooth_mean + std_err
        x_axis = np.arange(mean.shape[0])
        plt.fill_between(x_axis, under_line, over_line, alpha=.1)
        plt.plot(mean.T, linewidth=2)
        plt.show()

    def plot_gradients(self, layer_n: int):
        mean = self.history[:, layer_n, 1, 0]
        std_err = self.history[:, layer_n, 1, 1]
        smooth_mean = self._smooth_curve(mean)
        under_line = smooth_mean - std_err
        over_line = smooth_mean + std_err
        x_axis = np.arange(mean.shape[0])
        plt.fill_between(x_axis, under_line, over_line, alpha=.1)
        plt.plot(mean.T, linewidth=2)
        plt.show()

    def _smooth_curve(self, data):
        # smooth the a curve taking the average of the n last samples if
        # required
        avg_len = 100 # arbitrary
        avg_data = []
        for i in range(len(data)):  # iterate through rewards
            # get the previous rewards and the current one
            curr_data = data[i]
            last_n_data = [data[j] for j in range(i - avg_len - 1, i) 
                            if j >= 0]
            last_n_data.append(curr_data)
            # average the last n rewards
            avg_datum = np.average(last_n_data)
            avg_data += [avg_datum]
        return np.array(avg_data)


if __name__=="__main__":
    params = {
        "layers_info": [
        {"type": "linear", "input_size": 4, "output_size": 16, "activation": "relu"},
        {"type": "linear", "input_size": 16, "output_size": 2, "activation": "softmax"}
      ],
      "optimizer_info": {
        "type": "adam",
        "learning_rate": 0.0001
      }
    }
    set_random_seed(1)
    nn = CustomNeuralNetwork(params)
    #nn.set_seed(1)
    writer = SummaryWriter("/home/vaillus/projects/RL_toolkit/logs/actor_critic")
    input = torch.randn(4)
    
    writer.add_graph(nn, input)
    writer.close()
    print(nn(input))