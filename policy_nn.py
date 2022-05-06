from custom_nn import CustomNeuralNetwork
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, Dict


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const=0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight.data, std)
    #nn.init.constant_(layer.bias, bias_const) # in sb3, bias is still there
    return layer



class PolicyNetwork(CustomNeuralNetwork):
    def __init__(self,
        layers_info: Dict[str, Any],
        optimizer_info: Dict[str, Any],
        seed: Optional[int] = None,
        input_dim: int = 0,
        output_dim: int = 0,
        is_continuous: bool = False
    ):
        super().__init__(layers_info, optimizer_info, seed, input_dim, output_dim)
        self.is_continuous = is_continuous
        self.mu = None
        self.sigma = None
        if self.is_continuous:
            self.adapt_last_layer_to_continuous()

    def adapt_last_layer_to_continuous(self):
        """ removing last layer because we are going to replace it with 
        mu and sigma.
        """
        self.layers = self.layers[:-1] 
        # init mu
        self.mu = nn.Sequential(
            layer_init(nn.Linear(
                self._get_layers_sizes(self.layers_info["sizes"])[-1],
                self.output_dim
            ), std = 0.01),
            nn.Tanh()
        )
        # init sigma
        # case where std is state-dependant
        """self.sigma = nn.Linear(
                self._get_layers_sizes(self.layers_info["sizes"])[-1], 
                self.output_dim
            )"""
        self.sigma = nn.Parameter(torch.zeros(1, self.output_dim))
        torch.nn.init.orthogonal_(self.sigma, 0.01)
        #self.mu[0].weight.data *= 0.01
        #self.sigma.weight.data *= 0.01
    
    def forward(self, x):
        # format the input data
        #x = super().forward(x)
        if self.is_continuous:
            x = self._format_input(x)
            num_layers = len(self.layers)
            for i in range(num_layers):
                x = self.forward_layer(x,i)
            m = nn.Softplus() # only for state-dependant sigma
            #mu = self.mu(x)
            #logstd = 
            std = m(self.sigma -0.1879)
            return self.mu(x), std#m(self.sigma(x) -0.1879)
        else: 
            # discrete case
            return super().forward(x)
        
    def reinit_layers(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList() # erase previous layers
        self.activations = []
        self.optimizer = None
        self._init_layers(self.layers_info) # create new ones
        if self.is_continuous:
            self.adapt_last_layer_to_continuous()
        self._init_optimizer(self.optimizer_info)