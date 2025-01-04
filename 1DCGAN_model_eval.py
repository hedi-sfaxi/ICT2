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
    min_val = np.min(data, axis=0, keepdims=True)
    max_val = np.max(data, axis=0, keepdims=True)
    return (data - min_val) / (max_val - min_val) * 2 - 1


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
    def __init__(self, latent_dim, channels):
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
generator = Generator(latent_dim, channels).to(device)
discriminator = Discriminator(time_steps, channels).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Stack the three signals into a single tensor with shape (num_samples, time_steps, channels)
real_data = np.stack([de_data_normalized.flatten(), fe_data_normalized.flatten(), ba_data_normalized.flatten()], axis=-1)
real_data = torch.tensor(real_data, dtype=torch.float32).permute(1, 0).unsqueeze(0)  # (batch, channels, time_steps)

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


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Helper function to visualize signals
def plot_signals(real_signals, fake_signals):
    """
    Plots a subset of real and generated signals for comparison.

    Parameters:
    - real_signals: numpy array of real signals
    - fake_signals: numpy array of generated signals
    """
    num_signals = min(real_signals.shape[0], fake_signals.shape[0])
    print(f"Number of signals to plot: {num_signals}")
    
    plt.figure(figsize=(12, 8))
    for i in range(num_signals):
        plt.subplot(num_signals, 1, i + 1)
        plt.plot(real_signals[i], label="Real Signal")
        plt.plot(fake_signals[i], label="Generated Signal")
        plt.legend()
        plt.title(f"Comparison of Signal {i + 1}")
    plt.tight_layout()
    plt.show()


# Function to calculate statistical similarity
def compute_statistics(real_signals, fake_signals):
    real_mean = np.mean(real_signals, axis=0)
    real_std = np.std(real_signals, axis=0)
    fake_mean = np.mean(fake_signals, axis=0)
    fake_std = np.std(fake_signals, axis=0)

    print("Real Data Mean:", real_mean)
    print("Generated Data Mean:", fake_mean)
    print("Real Data Std Dev:", real_std)
    print("Generated Data Std Dev:", fake_std)

# Evaluate the Generator
def evaluate_generator(generator, real_data, latent_dim, device):
    # Generate fake signals
    noise = torch.randn(real_data.size(0), latent_dim, device=device)
    fake_signals = generator(noise).detach().cpu().numpy()

    # Real signals for comparison
    real_signals = real_data.cpu().numpy()

    # Debug: Print shapes
    print(f"Shape of generated signals before squeeze: {fake_signals.shape}")
    print(f"Shape of real signals before squeeze: {real_signals.shape}")

    # Adjust shapes for compatibility
    fake_signals = fake_signals.squeeze()  # Remove singleton dimensions
    real_signals = real_signals.squeeze()  # Remove singleton dimensions

    # Ensure correct transpose based on dimensions
    if fake_signals.ndim == 3:
        fake_signals = fake_signals.transpose(0, 2, 1)
    elif fake_signals.ndim == 2:
        fake_signals = fake_signals[np.newaxis, :, :]  # Add a singleton batch dimension

    if real_signals.ndim == 3:
        real_signals = real_signals.transpose(0, 2, 1)
    elif real_signals.ndim == 2:
        real_signals = real_signals[np.newaxis, :, :]  # Add a singleton batch dimension

    # Debug: Print shapes after adjustments
    print(f"Shape of fake signals after adjustments: {fake_signals.shape}")
    print(f"Shape of real signals after adjustments: {real_signals.shape}")

    # Truncate real signals to match the length of fake signals
    min_length = min(real_signals.shape[2], fake_signals.shape[2])
    real_signals_truncated = real_signals[:, :, :min_length]
    fake_signals_truncated = fake_signals[:, :, :min_length]

    # Debug: Print shapes after truncation
    print(f"Shape of truncated real signals: {real_signals_truncated.shape}")
    print(f"Shape of truncated fake signals: {fake_signals_truncated.shape}")

    # Visualize signals
    plot_signals(real_signals_truncated[0], fake_signals_truncated[0])

    # Compute and print statistics
    compute_statistics(real_signals_truncated, fake_signals_truncated)

    # PCA Visualization
    combined_signals = np.concatenate((real_signals_truncated, fake_signals_truncated), axis=0)
    combined_flattened = combined_signals.reshape(combined_signals.shape[0], -1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_flattened)
    plt.scatter(pca_result[:real_signals_truncated.shape[0], 0], pca_result[:real_signals_truncated.shape[0], 1], label="Real Signals")
    plt.scatter(pca_result[real_signals_truncated.shape[0]:, 0], pca_result[real_signals_truncated.shape[0]:, 1], label="Generated Signals")
    plt.legend()
    plt.title("PCA Visualization")
    plt.show()

    # t-SNE Visualization with Automatic Perplexity Adjustment
    combined_signals = np.concatenate((real_signals_truncated, fake_signals_truncated), axis=0)
    combined_flattened = combined_signals.reshape(combined_signals.shape[0], -1)

    # Determine the number of samples
    n_samples = combined_flattened.shape[0]

    # Adjust perplexity
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples

    if perplexity < 1:
        print("Not enough samples for t-SNE visualization.")
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(combined_flattened)

        # Plot the results
        plt.scatter(tsne_result[:real_signals_truncated.shape[0], 0], tsne_result[:real_signals_truncated.shape[0], 1], label="Real Signals")
        plt.scatter(tsne_result[real_signals_truncated.shape[0]:, 0], tsne_result[real_signals_truncated.shape[0]:, 1], label="Generated Signals")
        plt.legend()
        plt.title("t-SNE Visualization")
        plt.show()



# Evaluate the generator
evaluate_generator(generator, real_data, latent_dim, device)
