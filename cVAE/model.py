import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()

        self.input = nn.Linear(input_dim + label_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, label):
        out = self.input(torch.cat((x, label), -1))
        out = F.relu(out)
        out = self.hidden(out)
        out = F.relu(out)
        out = self.output(out)

        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()

        self.input = nn.Linear(latent_dim + label_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, label):
        out = self.input(torch.cat((z, label), -1))
        out = F.relu(out)
        out = self.hidden(out)
        out = F.relu(out)
        out = self.output(out)

        return out


class cVAE(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, label_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, label_dim, hidden_dim, input_dim)

    def forward(self, x, label):
        latent = self.encoder(x, label)
        out = self.decoder(latent, label)

        return out
