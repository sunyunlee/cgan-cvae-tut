from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

from model import cVAE

""" Organize dataset """
bos = load_boston()
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df["Price"] = bos.target

data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data["Price"] = df.Price

# Dataset to numpy
X = torch.tensor(data.drop("Price", axis=1).values)
Y = torch.tensor(data["Price"].values)

# Split dataset for test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=42)
Y_train = Y_train.view(-1, 1)
Y_test = Y_test.view(-1, 1)

""" Hyperparameter """
N_EPOCHS = 50
BATCH_SIZE = 64
lr = 0.001
INPUT_DIM = X_train.shape[1]
LABEL_DIM = Y_train.shape[1]
LATENT_DIM = 5
HIDDEN_DIM = 20


""" Loss function """
def loss_fn(mu_z, std_z, z_sample, mu_x, std_x, x_sample):
    S = x_sample.shape[0]

    # log posterior q(z|x)
    q_z_dist = torch.distributions.Normal(mu_z, std_z)
    log_q_z = q_z_dist.log_prob(z_sample)

    # log likelihood p(x|z)
    p_x_dist = torch.distributions.Normal(mu_x, std_x)
    log_p_x = p_x_dist.log_prob(x_sample)

    # log prior p(z)
    p_z_dist = torch.distributions.Normal(0, 1)
    log_p_z = p_z_dist.log_prob(z_sample)

    loss = (1 / S) * (
        torch.sum(log_q_z) - torch.sum(log_p_x) - torch.sum(log_p_z)
    )

    return loss


""" Train the model """
train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                         shuffle=True)

# Define the model
model = cVAE(INPUT_DIM, LABEL_DIM, HIDDEN_DIM, LATENT_DIM).type(torch.float64)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(N_EPOCHS):
    for x, y in train_iter:
        # Update the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        mu_z, std_z, z_sample, mu_x, std_x, x_sample = model(x, y)

        # Loss
        loss = loss_fn(mu_z, std_z, z_sample, mu_x, std_x, x_sample)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    print("Train Epoch: {} Loss: {}".format(ep + 1, loss))


""" Test """
test_data = torch.utils.data.TensorDataset(X_test, Y_test)
test_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                         shuffle=True)
for i, (x, y) in enumerate(test_iter):
    # Forward pass
    mu_z, std_z, z_sample, mu_x, std_x, x_sample = model(x, y)

    # Loss
    loss = loss_fn(mu_z, std_z, z_sample, mu_x, std_x, x_sample)

    print("Test Epoch: {} Loss: {}".format(i + 1, loss))
