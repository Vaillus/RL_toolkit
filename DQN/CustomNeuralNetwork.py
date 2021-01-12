import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CustomNeuralNetwork(nn.Module):
    """
    Attempt to make an easy-to-use neural net class
    """
    def __init__(self, params):
        super(CustomNeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = []
        self.init_layers(params["layers_info"])
        self.optimizer = None
        self.init_optimizer(params["optimizer_info"])
        #optim.Adam(self.parameters(), lr=learning_rate)

    def init_layers(self, layers_info):
        for layer_info in layers_info:
            if layer_info["type"] == "linear":
                layer = nn.Linear(layer_info["input_size"], layer_info["output_size"])
                layer.weight.data.normal_(0, 0.1) # TODO : I don't think it should be optional, but it might be. See later.
            self.layers.append(layer)
            self.activations.append(layer_info["activation"])

    def init_optimizer(self, optimizer_info):
        if optimizer_info["type"] == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=optimizer_info["learning_rate"])

    def forward(self, x):
        # format the input data
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
        elif isinstance(x, list):
            x = torch.FloatTensor(x)
        elif isinstance(x, tuple):
            x = torch.FloatTensor(x)

        for i in range(len(self.layers)):
            if self.activations[i] == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activations[i] == "tanh":
                x = F.tanh(self.layers[i](x))
            elif self.activations[i] == "softmax":
                x = F.softmax(self.layers[i](x))
            else:
                x = self.layers[i](x)

        return x

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()  # il faut que la loss ait une seule valeur.
        self.optimizer.step()

if __name__=="__main__":
    params = {
        "layers_info": [
            {"type": "linear", "input_size": 30, "output_size": 3, "activation": "relu"},
            {"type": "linear", "input_size": 3, "output_size": 1, "activation": "tanh"}
        ],
        "optimizer_info": {
            "type": "adam",
            "learning_rate": 0.000001
        }
    }
    nn = CustomNeuralNetwork(params)
    input = torch.randn(30)

    print(nn(input))