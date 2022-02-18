import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
        self.layers_info = layers_info # for reinit purposes
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self._init_layers(layers_info)
        self._init_optimizer(optimizer_info)
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
    


    # === initialization functions =====================================



    def set_seed(self, seed:int) -> None:
        """ Set a seed for the neural net and reinit layers with the new 
        seed"""
        self.seed = seed
        set_random_seed(seed)
        # reinit the layers initial values
        for i in range(len(self.layers)):
            self.layers[i].weight.data.normal_(0, 0.1)

    def _init_layers(self, layers_info:Dict[str, Any]) -> None:
        """get and format the parameters from the layers info.
        Init the layers from this.

        Args:
            layers_info (Dict[str, Any])
        """
        # extract variables from the input dictionary
        n_hid_layers, types, sizes, activations = self._extract_layer_var(layers_info)
        # initialize the layers with the variables
        for i in range(n_hid_layers + 1):
            if types[i] == "linear":
                # initialize the layer with the correct size and add it 
                # to the list
                input_size = self._get_input_size(i, sizes)
                output_size = self._get_output_size(i, sizes, n_hid_layers)
                layer = nn.Linear(input_size, output_size)
                self.layers.append(layer)
        self.activations = activations

    def _get_input_size(self, i, sizes):
        if i == 0:
            return self.input_dim
        else:
            return sizes[i-1]

    def _get_output_size(self, i, sizes, n_hid_layers):
        if i == n_hid_layers:
            return self.output_dim
        else:
            return sizes[i]

    def _extract_layer_var(self,layers_info):
        """make a tuple of variables from a dictionary containing layers 
        information"""
        n_hid_layers = layers_info["n_hidden_layers"]
        types = self._get_layers_types(layers_info["types"], n_hid_layers)
        sizes = self._get_layers_sizes(layers_info["sizes"], n_hid_layers)
        activations = self._get_layers_activations(
            layers_info["hidden_activations"],
            layers_info["output_activation"],
            n_hid_layers)
        
        return n_hid_layers, types, sizes, activations

    def _get_layers_types(self, param_types, n_hid_layers):
        # types of the layers
        if isinstance(param_types, list):
            assert len(param_types) == n_hid_layers + 1
            types = param_types
        else:
            types = [param_types] * (n_hid_layers + 1)
        return types
    
    def _get_layers_sizes(self, param_sizes, n_hid_layers):
        # sizes of the layers
        if isinstance(param_sizes, list):
            assert len(param_sizes) == n_hid_layers
            sizes = param_sizes
        else:
            sizes = [param_sizes] * n_hid_layers
        return sizes

    def _get_layers_activations(
        self, 
        hidden_activations, 
        output_activation,
        n_hid_layers
    ):
        # activation functions of the layers
        if isinstance(hidden_activations, list):
            assert len(hidden_activations) == n_hid_layers
            activations = hidden_activations
        else:
            activations = [hidden_activations] * (n_hid_layers)
        activations += [output_activation]
        return activations

    def _init_optimizer(self, optimizer_info):
        if optimizer_info["type"] == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=optimizer_info["learning_rate"])



    # === other functions ==============================================



    def forward(self, x):
        # format the input data
        x = self._format_input(x)
        num_layers = len(self.layers)
        for i in range(num_layers):
            x = self.forward_layer(x,i)

        return x

    # TODO: If I don't get in trouble for commenting this, delete it.
    #def embedding(self, x):
    #    x = self._format_input(x)
    #    num_layers = len(self.layers)
    #    # get the point associated with the input in the latent space of the last layer
    #    for i in range(num_layers-1):
    #        x = self.forward_layer(x,i)
    #    return x
    
    #def get_embedding_size(self):
    #    return self.layers[-2].out_features

    def _format_input(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, list):
            x = torch.Tensor(x)
        elif isinstance(x, tuple):
            x = torch.Tensor(x)
        x = x.float()
        return x

    def forward_layer(self, x, i):
        if self.activations[i] == "relu":
            x = torch.relu(self.layers[i](x))
        elif self.activations[i] == "tanh":
            x = torch.tanh(self.layers[i](x))
        elif self.activations[i] == "softmax":
            x = F.softmax(self.layers[i](x), dim= -1)
        elif self.activations[i] == "none":
            x = self.layers[i](x)
        else:
            x = self.layers[i](x)
        return x

    def backpropagate(self, loss):
        """update the weight with the gradients and step the optimizer"""
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