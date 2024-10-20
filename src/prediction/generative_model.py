# src/prediction/generative_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GenerativePredictor:
    def __init__(self, data):
        self.data = data
    
    def train_model(self, epochs=50, batch_size=64):
        input_dim = self.data.shape[1]
        model = VAE(input_dim, 128, 20)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(torch.tensor(self.data.values, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_idx, (data_batch,) in enumerate(train_loader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data_batch)
                recon_loss = nn.functional.mse_loss(recon_batch, data_batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}")
        self.model = model
    
    def generate_predictions(self, num_samples=1):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 20)
            samples = self.model.decode(z)
        return samples.numpy()
