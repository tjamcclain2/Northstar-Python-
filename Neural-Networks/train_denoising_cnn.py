# Code to Train a Neural Network from Preprocessed GW Data
"""
Created on Thu Jul 17 09:35:36 2025

@author: nickr
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# --- Custom Dataset ---
class GWTrainingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.spectrograms = sorted([f for f in os.listdir(data_dir) if f.startswith("X_input")])
        self.waveforms = sorted([f for f in os.listdir(data_dir) if f.startswith("y_target")])

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec_path = os.path.join(self.data_dir, self.spectrograms[idx])
        wave_path = os.path.join(self.data_dir, self.waveforms[idx])
        spec = np.load(spec_path)[None, :, :]  # Add channel dimension
        wave = np.load(wave_path)
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(wave, dtype=torch.float32)

# --- Neural Network ---
class SpectrogramToWaveformNet(nn.Module):
    def __init__(self, input_shape, output_length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            flat_size = self.encoder(dummy).view(1, -1).shape[1]
        self.decoder = nn.Sequential(
            nn.Linear(flat_size, 1024), nn.ReLU(),
            nn.Linear(1024, output_length)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)

# --- Load Data ---
data_dir = "C:/Users/nickr/Downloads/MyGWData"
dataset = GWTrainingDataset(data_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# --- Initialize Model ---
example_input, example_output = dataset[0]
input_shape = example_input.shape[1:]
output_length = example_output.shape[0]

model = SpectrogramToWaveformNet(input_shape, output_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
num_epochs = 20
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

# --- Save Model and Plot Loss ---
torch.save(model.state_dict(), os.path.join(data_dir, "denoising_cnn_model.pt"))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.title("Training and Validation Loss")
plt.savefig(os.path.join(data_dir, "training_loss_plot.png"))
plt.show()
print(f"Found {len(os.listdir(data_dir))} files in data directory.")

