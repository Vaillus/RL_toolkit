import torch
import pandas as pd
from sklearn.datasets import load_iris
# import a dataset to make a logistic regression model
from torch.utils.data import TensorDataset, DataLoader

import CustomNeuralNetwork as cnn
import wandb

wandb.init(job_type="training")

# load iris dataset
iris = load_iris(as_frame=True)
# return x y = True
x, y = iris.data, iris.target
print(y)



params = {
    "layers_info": {
        "input_dim": 4,
        "output_dim": 1,
        "n_hidden_layers": 1,
        "types": "linear",
        "sizes": 4,
        "hidden_activations": "none",
        "output_activation": "relu"
    },
    "optimizer_info": {
        "type": "adam",
        "learning_rate": 0.05
    }
}

l1 = torch.nn.Linear(4, 1)
torch.nn.init.xavier_normal_(l1.weight)
act1 = torch.nn.ReLU()
opt = torch.optim.Adam(l1.parameters(), lr=0.05)

x = torch.Tensor(x[y != 2].values)
x = (x - x.mean()) / (x.std() + 1e-8)

y = torch.Tensor(y[y != 2].values)

Benoit = torch.nn.BCELoss()

def Hugo(x, y):
    return torch.mean(torch.abs(((torch.sign(x - 1e-26) + 1) / 2) - y))

wandb.watch(l1,log_freq=1)
MSELoss = torch.nn.MSELoss()
n_iter= 1000
for i in range(n_iter):
    opt.zero_grad()
    pred = act1(l1(x))

    loss = Hugo(pred.squeeze(), y)
    loss.backward()
    opt.step()
    wandb.log({"loss": loss.item()})
    print(loss.item())