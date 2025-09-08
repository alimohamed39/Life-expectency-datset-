import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas
import numpy as np



class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Linear(19,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )

    def forward(self, x):
        return self.net(x)