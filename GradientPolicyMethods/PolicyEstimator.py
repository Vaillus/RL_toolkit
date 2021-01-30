import torch
import torch.nn as nn
import torch.optim as optim


class PolicyEstimator():
    def __init__(self, params):
        self.input_size = params.get("input_size", 2)
        self.output_size = params.get("output_size", 2)
        self.α = params.get("learning_rate", 0.9)
        self.is_continuous = params.get("is_continuous", False)  # to be used later

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            #nn.Tanh())
            nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.α)

    def predict(self, state):
        """ gives the probability of actions under current policy
        """
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs
