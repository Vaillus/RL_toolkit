import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import set_random_seed
from typing import Optional, Any, Dict


class CustomNeuralNetwork(nn.Module):
    """ Attempt to make an easy-to-use neural net class
    """
    def __init__(
        self,
        layers_info: Dict[str, Any],
        optimizer_info: Dict[str, Any],
        seed: Optional[int] = None,
        input_dim: int = 0,
        output_dim: int = 0
    ):
        super(CustomNeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = []
        self.optimizer = None
        self.optimizer_info = optimizer_info
        self.seed = None
        self.history = np.array([]) # TODO: get rid of it. Don't need it anymore.
        self.layers_info = layers_info # for reinit purposes
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self._init_layers(layers_info)
        self._init_optimizer(optimizer_info)
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
    
    def set_seed(self, seed):
        self.seed = seed
        set_random_seed(seed)
        for i in range(len(self.layers)):
            self.layers[i].weight.data.normal_(0, 0.1)

    def _init_layers(self, layers_info):
        # check dimensions of parameters
        n_hid_layers = layers_info["n_hidden_layers"]
        if isinstance(layers_info["types"], list):
            assert len(layers_info["types"]) == n_hid_layers + 1
            types = layers_info["types"]
        else:
            types = [layers_info["types"]] * (n_hid_layers + 1)
        if isinstance(layers_info["sizes"], list):
            assert len(layers_info["sizes"]) == n_hid_layers
            sizes = layers_info["sizes"]
        else:
            sizes = [layers_info["sizes"]] * n_hid_layers
        if isinstance(layers_info["activations"], list):
            assert len(layers_info["activations"]) == n_hid_layers + 1
            activations = layers_info["activations"]
        else:
            activations = [layers_info["activations"]] * (n_hid_layers + 1)
        # TODO handle flatten when the input_size is a list.
        # layers initialization
        for i in range(n_hid_layers + 1):
            if types[i] == "linear":
                if i == 0:
                    input_size = self.input_dim
                else:
                    input_size = sizes[i-1]
                if i == n_hid_layers:
                    output_size = self.output_dim
                else:
                    output_size = sizes[i]
                layer = nn.Linear(input_size, output_size)
                self.layers.append(layer)
        self.activations = activations
        # old way of doing it
        """for layer_info in layers_info:
            if layer_info["type"] == "linear":
                layer = nn.Linear(layer_info["input_size"], layer_info["output_size"])
                layer.weight.data.normal_(0, 0.1)  
            
            self.layers.append(layer)
            # next line could work. It is useful mainly on a policy network, apparently.
            #self.layers[-1].weight.data *= 0.01
        """

    def _init_optimizer(self, optimizer_info):
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
                x = torch.relu(self.layers[i](x))
            elif self.activations[i] == "tanh":
                x = torch.tanh(self.layers[i](x))
            elif self.activations[i] == "softmax":
                x = F.softmax(self.layers[i](x))
            elif self.activations[i] == "none":
                x = self.layers[i](x)
            else:
                x = self.layers[i](x)

        return x

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print("coucou")
    
    def reinit_layers(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList() # erase previous layers
        self.activations = []
        self.optimizer = None
        self._init_layers(self.layers_info) # create new ones
        self._init_optimizer(self.optimizer_info)

    

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
    input = torch.randn(4)
    print(nn(input))