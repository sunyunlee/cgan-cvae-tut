import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, label_dim):
        super().__init__()

        self.input = nn.Linear(noise_dim + label_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, noise, label):
        out = self.input(torch.cat((noise, label), -1))
        out = F.relu(out)
        out = self.hidden(out)
        out = F.relu(out)
        out = self.output(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim):
        super().__init__()

        self.input = nn.Linear(input_dim + label_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, label):
        out = self.input(torch.cat((x, label), -1))
        out = F.relu(out)
        out = self.hidden(out)
        out = F.relu(out)
        out = self.output(out)
        out = F.sigmoid(out)

        return out

