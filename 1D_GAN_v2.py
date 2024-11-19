import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

## Access the Variables 

# Path to your .mat file
file_path = r'C:\Users\yasmi\Desktop\Projet ICT 2\Code\CRWUdata\CRWU\12k Drive End Bearing Fault Data\Ball\0007\B007_0.mat'

# Load the .mat file
mat_data = scipy.io.loadmat(file_path)

# Access the time series data for each dimension
de_data = mat_data['X118_DE_time']  # Drive End data
fe_data = mat_data['X118_FE_time']  # Fan End data
ba_data = mat_data['X118_BA_time']  # Base Acceleration data

# Print the shape of each dataset to confirm
print(f"Drive End data shape: {de_data.shape}")
print(f"Fan End data shape: {fe_data.shape}")
print(f"Base Acceleration data shape: {ba_data.shape}")

## Prepocess the data 

# Normalize the data to be between -1 and 1 for GAN training
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1

# Normalize each dimension
de_data_normalized = normalize_data(de_data)
fe_data_normalized = normalize_data(fe_data)
ba_data_normalized = normalize_data(ba_data)

# Stack the data along the last axis to combine the dimensions
time_series_data = np.stack([de_data_normalized, fe_data_normalized, ba_data_normalized], axis=-1)

# Display the shape of the stacked data
print(f"Stacked data shape: {time_series_data.shape}")

# Reshape the data to remove the singleton dimension
time_series_data = time_series_data.reshape(122571, 3)  # Now it should be (122571, 3)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100   # Size of the latent vector (input to the Generator)
time_steps = 122571  # Number of time steps (length of the time series)
channels = 3        # Number of channels (e.g., DE, FE, BA)
batch_size = 32
num_epochs = 1000
learning_rate = 0.0002
beta1 = 0.5

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, time_steps, channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, 4, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # Reshape input to match input dimensions
        z = z.view(z.size(0), latent_dim, 1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, time_steps, channels):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(channels, 32, 4, 2, 1),  # Downsample: 122571 -> 61285
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 4, 2, 1),        # Downsample: 61285 -> 30642
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),       # Downsample: 30642 -> 15321
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Add global pooling to reduce the feature map to a single value per batch
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, 128, 1)

        # Final classifier layer
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten to (batch_size, 128)
            nn.Linear(128, 1),  # Fully connected layer to scalar output
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.global_pool(x)  # Perform global pooling
        return self.classifier(x)  # Shape: (batch_size,)



# Initialize models
generator = Generator(latent_dim, time_steps, channels).to(device)
discriminator = Discriminator(time_steps, channels).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Stack the three signals into a single tensor with shape (num_samples, time_steps, channels)
real_data = np.stack([de_data.flatten(), fe_data.flatten(), ba_data.flatten()], axis=-1)  # Shape: (time_steps, channels)
real_data = torch.tensor(real_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension: (1, time_steps, channels)

# For PyTorch, reorder to (batch_size, channels, time_steps)
real_data = real_data.permute(0, 2, 1)  # Shape: (batch_size, channels, time_steps)

# Create DataLoader
dataset = TensorDataset(real_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # Extract real samples
        real_samples = data[0].to(device)  # Shape: (batch_size, channels, time_steps)

        # Real labels
        batch_size = real_samples.size(0)
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Forward pass with real data
        real_output = discriminator(real_samples)  # Output is (batch_size,)
        d_loss_real = criterion(real_output, real_labels)

        # Generate fake samples
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(z)
        fake_output = discriminator(fake_samples.detach())  # Detach to avoid updating generator during D training
        d_loss_fake = criterion(fake_output, fake_labels)

        # Total Discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_samples)
        g_loss = criterion(fake_output, real_labels)  # We want the generator to fool the discriminator
        g_loss.backward()
        optimizer_G.step()

    # Print losses
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

print("Training finished!")