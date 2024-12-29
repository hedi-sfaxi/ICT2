import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch._dynamo
torch._dynamo.config.skip_fsdp_hooks = True
from evaluation import kl_divergence
import torch.optim as optim


class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(TimeGAN, self).__init__()
        self.device = device

        # Encoder
        self.encoder = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.RNN(hidden_dim, input_dim, num_layers, batch_first=True)

        # Generator
        self.generator = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.generator_fc = nn.Linear(hidden_dim, input_dim)

        # Discriminator
        self.discriminator = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.discriminator_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return encoded, decoded

    def generate(self, z):
        generated, _ = self.generator(z)
        return self.generator_fc(generated)

    def discriminate(self, x):
        out, _ = self.discriminator(x)
        return torch.sigmoid(self.discriminator_fc(out))
    
    

def train_time_gan(model, data, batch_size, num_epochs, device, save_path):
    optimizer_gen = optim.Adam(
        list(model.generator.parameters()) + list(model.generator_fc.parameters()), lr=0.001
    )
    optimizer_dis = optim.Adam(
        list(model.discriminator.parameters()) + list(model.discriminator_fc.parameters()), lr=0.001
    )

    
    # Variable pour suivre la meilleure kl Divergence
    criterion = nn.BCELoss()

    best_kl_divergence = float('inf')
    for epoch in range(num_epochs):
        kl_divergences = []

        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            batch = torch.tensor(batch, dtype=torch.float32).to(device)

            # Entraînement du Discriminateur
            z = torch.randn(batch.size(0), batch.size(1), batch.size(2)).to(device)
            generated_data = model.generate(z)

            real_labels = torch.ones(batch.size(0), batch.size(1), 1).to(device)
            fake_labels = torch.zeros(batch.size(0), batch.size(1), 1).to(device)

            dis_real = model.discriminate(batch)
            dis_fake = model.discriminate(generated_data.detach())

            loss_dis = criterion(dis_real, real_labels) + criterion(dis_fake, fake_labels)

            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            # Entraînement du Générateur
            dis_fake = model.discriminate(generated_data)
            loss_gen = criterion(dis_fake, real_labels)

            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
            
            kl_div = kl_divergence(batch, generated_data)
            kl_divergences.append(kl_div)

        avg_kl_div = np.nanmean(kl_divergences)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_dis.item()}, Loss G: {loss_gen.item()}) KL divergence: {avg_kl_div}")

        # Sauvegarde du modèle si la KL Divergence est meilleure
        if avg_kl_div < best_kl_divergence:
            best_kl_divergence = avg_kl_div
            print(f"New best kl Divergence: {best_kl_divergence}. Saving model...")

            # Sauvegarder le modèle
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_dis_state_dict': optimizer_dis.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")