import torch
import torch.nn as nn
import torch.optim as optim
from utils import extract_time, rnn_cell, random_generator, batch_generator
from tqdm import tqdm

def timegan(ori_data, parameters):
    """TimeGAN function.

    Use original data as training set to generate synthetic data (time-series).

    Args:
        - ori_data: original time-series data
        - parameters: TimeGAN network parameters

    Returns:
        - generated_data: generated time-series data
    """
    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = torch.min(data, dim=0, keepdim=True)[0]
        data = data - min_val
        max_val = torch.max(data, dim=0, keepdim=True)[0]
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    class Embedder(nn.Module):
        def __init__(self):
            super(Embedder, self).__init__()
            self.rnn = nn.RNN(input_size=dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            outputs, _ = self.rnn(x)
            return torch.sigmoid(self.fc(outputs))

    class Recovery(nn.Module):
        def __init__(self):
            super(Recovery, self).__init__()
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, dim)

        def forward(self, h):
            outputs, _ = self.rnn(h)
            return torch.sigmoid(self.fc(outputs))

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.rnn = nn.RNN(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, z):
            outputs, _ = self.rnn(z)
            return torch.sigmoid(self.fc(outputs))

    class Supervisor(nn.Module):
        def __init__(self):
            super(Supervisor, self).__init__()
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers - 1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h):
            outputs, _ = self.rnn(h)
            return torch.sigmoid(self.fc(outputs))

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h):
            outputs, _ = self.rnn(h)
            return self.fc(outputs)

    # Initialize networks
    embedder = Embedder()
    recovery = Recovery()
    generator = Generator()
    supervisor = Supervisor()
    discriminator = Discriminator()

    # Optimizers
    e_optimizer = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()))
    g_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()))
    d_optimizer = optim.Adam(discriminator.parameters())

     # Optimizers
    e_optimizer = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()))
    g_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()))
    d_optimizer = optim.Adam(discriminator.parameters())

    history = {
        'e_loss_t0': [],
        'G_loss_S': [],
        'G_loss': [],
        'D_loss': []
    }

    # Phase 1: Train Embedder and Recovery
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = torch.tensor(X_mb, dtype=torch.float32)

        # Forward pass
        H = embedder(X_mb)
        X_tilde = recovery(H)

        # Compute reconstruction loss
        e_loss_t0 = nn.MSELoss()(X_mb, X_tilde)

        # Backward pass
        e_optimizer.zero_grad()
        e_loss_t0.backward(retain_graph=True)
        e_optimizer.step()

        history['e_loss_t0'].append(e_loss_t0.item())
        if itt % 1000 == 0:
            print(f"Step: {itt}/{iterations}, e_loss_t0: {e_loss_t0.item():.4f}")

    print("Finish Embedding Network Training")

    # Phase 2: Train Generator and Supervisor
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = torch.tensor(random_generator(batch_size, z_dim, T_mb, max_seq_len), dtype=torch.float32)

        H_fake = generator(Z_mb)
        H_supervise = supervisor(H_fake)

        # Supervised loss
        G_loss_S = nn.MSELoss()(H_fake[:, 1:, :], H_supervise[:, :-1, :])

        g_optimizer.zero_grad()
        G_loss_S.backward(retain_graph=True)
        g_optimizer.step()

        history['G_loss_S'].append(G_loss_S.item())
        if itt % 1000 == 0:
            print(f"Step: {itt}/{iterations}, G_loss_S: {G_loss_S.item():.4f}")

    print("Finish Training with Supervised Loss Only")

    # Phase 3: Joint Training
    for itt in range(iterations):
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = torch.tensor(random_generator(batch_size, z_dim, T_mb, max_seq_len), dtype=torch.float32)

            H_fake = generator(Z_mb)
            H_supervise = supervisor(H_fake)
            X_hat = recovery(H_supervise)

            # Unsupervised loss
            G_loss_U = nn.BCEWithLogitsLoss()(discriminator(H_supervise), torch.ones_like(discriminator(H_supervise)))

            # Feature matching loss
            G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0) + 1e-6) - torch.sqrt(X_mb.var(dim=0) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs(X_hat.mean(dim=0) - X_mb.mean(dim=0)))
            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = G_loss_U + gamma * G_loss_V

            g_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            g_optimizer.step()

        Y_real = discriminator(embedder(X_mb).detach())
        Y_fake = discriminator(H_fake.detach())

        D_loss_real = nn.BCEWithLogitsLoss()(Y_real, torch.ones_like(Y_real))
        D_loss_fake = nn.BCEWithLogitsLoss()(Y_fake, torch.zeros_like(Y_fake))
        D_loss = D_loss_real + D_loss_fake

        d_optimizer.zero_grad()
        D_loss.backward()
        d_optimizer.step()

        history['D_loss'].append(D_loss.item())
        history['G_loss'].append(G_loss.item())
        if itt % 1000 == 0:
            print(f"Step: {itt}/{iterations}, D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")

    print("Finish Joint Training")

    # Generate synthetic data
    Z_mb = torch.tensor(random_generator(no, z_dim, ori_time, max_seq_len), dtype=torch.float32)
    with torch.no_grad():
        generated_data_curr = recovery(supervisor(generator(Z_mb))).numpy()

    generated_data = [generated_data_curr[i, :ori_time[i], :] for i in range(no)]
    generated_data = [data * max_val + min_val for data in generated_data]

    return generated_data, history
