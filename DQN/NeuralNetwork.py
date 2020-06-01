import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    """
    Class that is specific to my first DQN use. I will probably get rid of it.
    """
    def __init__(self, input_size, output_size, learning_rate):
        super(Net, self).__init__()
        # TODO : handle input and output size
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 50)  # 6*6 from image dimension
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(30, output_size)
        self.fc3.weight.data.normal_(0, 0.1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)

        return x

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()  # il faut que loss ait une seule valeur.
        self.optimizer.step()



if __name__ == "__main__":
    n_inputs = 2
    n_actions = 3
    policy_net = Net()
    #policy_net.forward([0,0])
    #target_net = DQN(screen_height, screen_width, n_actions).to(device)
    net = Net()
    print(net)
    input1 = torch.tensor([[0,0], [1,1], [2,2]], dtype=torch.float32)
    input1 = torch.tensor([[0, 0]], dtype=torch.float64)
    output = net(input1)
    print(output)