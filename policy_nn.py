from custom_nn import CustomNeuralNetwork
import torch.nn as nn
from typing import Optional, Any, Dict

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
        self.std = None
        if self.is_continuous:
            self.adapt_last_layer_to_continuous()

    def adapt_last_layer_to_continuous(self):
        self.layers = self.layers[:-1]
        self.mu = nn.Sequential(
            nn.Linear(
                self._get_layers_sizes(self.layers_info["sizes"])[-1],
                self.output_dim
            ),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(
                self._get_layers_sizes(self.layers_info["sizes"])[-1], 
                self.output_dim
            ),
            nn.Softplus()
        )
    
    def forward(self, x):
        # format the input data
        #x = super().forward(x)
        if self.is_continuous:
            x = self._format_input(x)
            num_layers = len(self.layers)
            for i in range(num_layers):
                x = self.forward_layer(x,i)
            return self.mu(x), self.std(x)
        else: 
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