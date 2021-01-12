import torch
import torch.nn as nn
import torch.optim as optim


class BaselineNetwork():
    def __init__(self, params):
        self.input_size = params.get("input_size", 2)
        self.output_size = params.get("output_size", 1)
        self.learning_rate = params.get("learning_rate", 0.9)
        self.is_continuous = params.get("is_continuous", False)  # to be used later

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size)#,
            #nn.ReLU()
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs
