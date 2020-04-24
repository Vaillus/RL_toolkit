import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, 20)  # 6*6 from image dimension
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001) # TODO: lui il fait RMSprop

    def forward(self, x):
        self.optimizer.zero_grad()
        x = torch.from_numpy(x).to(torch.float32)
        print(f'input of the NN: {x}')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        print(f'output of the NN: {x}')
        return x

    def backpropagate(self, loss):
        #print(loss)
        loss.backward() # il faut que loss ait une seule valeur.
        """
        grads = get_gradient(one_hot(self.last_state, self.num_states), self.weights)

        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        self.weights = self.optimizer.update_weights(self.weights, g)
        """
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