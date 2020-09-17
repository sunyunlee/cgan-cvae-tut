from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from beeprint import pp
import pandas as pd
import numpy as np
import torch

from model import Generator, Discriminator


# Load dataset into DataFrame
bos = load_boston()
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df["Price"] = bos.target

# Standardize
data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price

# Dataset to numpy
X = torch.tensor(data.drop("Price", axis=1).values)
Y = torch.tensor(data["Price"].values)

# Split dataset for test and train
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=42)

Y_train = Y_train.view(-1, 1)
Y_test = Y_test.view(-1, 1)

# Define hyperparameters
N_EPOCHS = 50
BATCH_SIZE = 64
lr = 0.001
INPUT_DIM = X_train.shape[1]
LABEL_DIM = Y_train.shape[1]
NOISE_DIM = 5
HIDDEN_DIM = 20

train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                         shuffle=True)

# Define the model
generator = Generator(INPUT_DIM, HIDDEN_DIM, NOISE_DIM, LABEL_DIM).type(
    torch.float64)
discriminator = Discriminator(INPUT_DIM, HIDDEN_DIM, LABEL_DIM).type(
    torch.float64)

prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

loss_func = torch.nn.BCELoss()
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)

for ep in range(N_EPOCHS):
    for x, y in train_iter:
        N = len(x)
        """ Train Discriminator """
        ## Train with real data ##
        # Update the gradients to zero
        optimizerD.zero_grad()

        # Forward pass
        prob_out = discriminator(x, y)
        prob_real = torch.ones((N, 1)).type(torch.float64)

        # Loss
        lossD = loss_func(prob_out, prob_real)

        # Backward pass
        lossD.backward()

        # Update the weights
        optimizerD.step()

        ## Train with generated data ##
        # Update the gradients to zero
        optimizerD.zero_grad()

        # Forward pass on generator
        noise = prior.sample((N, NOISE_DIM)).type(torch.float64)
        x_gen = generator(noise, y)

        # Forward pass on discriminator
        prob_out = discriminator(x_gen, y)
        prob_real = torch.zeros((N, 1)).type(torch.float64)

        # Loss
        lossD_G = loss_func(prob_out, prob_real)

        # Backward pass
        lossD_G.backward()

        # Update the weights
        optimizerD.step()

        """ Train Generator """
        # Update the gradients to zero
        optimizerG.zero_grad()

        # Forward pass on generator
        # TODO: find out if I should use the same noise as above or sample
        #  new noise
        noise = prior.sample((N, NOISE_DIM)).type(torch.float64)
        x_gen = generator(noise, y)

        # Forward pass on discriminator
        prob_out = discriminator(x_gen, y)
        prob_real = torch.ones((N, 1)).type(torch.float64)

        # Loss
        lossG = loss_func(prob_out, prob_real)

        # Backward pass
        lossG.backward()

        # Update the weights
        optimizerG.step()

    print("Train Epoch: {} LossD: {} LossD_G: {} LossG: {}"
          .format(ep + 1, lossD, lossD_G, loss_G))

""" Test """
test_data = torch.utils.data.TensorDataset(X_test, Y_test)
test_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                         shuffle=True)
for i, (x, y) in enumerate(test_iter):
    N = len(x)
    """ Testing the discriminator """
    ## Test with real data ##
    # Forward pass
    prob_out = discriminator(x, y)
    prob_real = torch.ones((N, 1)).type(torch.float64)

    # Loss
    lossD = loss_func(prob_out, prob_real)

    ## Test with generated data ##
    # Forward pass on generator
    noise = prior.sample((N, NOISE_DIM)).type(torch.float64)
    x_gen = generator(noise, y)

    # Forward pass on discriminator
    prob_out = discriminator(x_gen, y)
    prob_real = torch.zeros((N, 1)).type(torch.float64)

    # Loss
    lossD_G = loss_func(prob_out, prob_real)

    """ Testing the generator """
    noise = prior.sample((N, NOISE_DIM)).type(torch.float64)
    x_gen = generator(noise, y)

    lossG = torch.nn.MSELoss(x_gen, x)

    print("Test Epoch: {} LossD: {} LossD_G: {} LossG: "
          .format(i + 1, lossD,  lossD_G, lossG))
