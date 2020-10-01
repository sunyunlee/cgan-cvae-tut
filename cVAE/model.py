import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()

        self.input = nn.Linear(input_dim + label_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mu_z = nn.Linear(hidden_dim, latent_dim)
        self.std_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, label):
        out = self.input(torch.cat((x, label), -1))
        out = self.bn1(out)
        out = F.relu(out)
        out = self.hidden(out)
        out = self.bn2(out)
        out = F.relu(out)
        mu_z = self.mu_z(out)
        std_z = self.std_z(out)

        return mu_z, std_z


class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()

        self.input = nn.Linear(latent_dim + label_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mu_x = nn.Linear(hidden_dim, output_dim)
        self.std_x = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, label):
        out = self.input(torch.cat((z, label), -1))
        out = self.bn1(out)
        out = F.relu(out)
        out = self.hidden(out)
        out = self.bn2(out)
        out = F.relu(out)
        mu_x = self.mu_x(out)
        std_x = self.std_x(out)

        return mu_x, std_x

class cVAE(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, label_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, label_dim, hidden_dim, input_dim)

    def forward(self, x, label):
        # Encoder
        mu_z, std_z = self.encoder(x, label)

        # Sample z
        eps = torch.randn_like(std_z)
        z_samples = mu_z + eps * torch.exp(std_z)

        # Decoder
        mu_x, std_x = self.decoder(z_samples, label)
        eps = torch.randn_like(std_x)

        x_samples = mu_x + eps * torch.exp(std_x)

        return mu_z, std_z, z_samples, mu_x, std_x, x_samples
