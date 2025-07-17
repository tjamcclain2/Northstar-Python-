#Code to test Neural Network with a real GW event
import os
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from gwpy.timeseries import TimeSeries
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, spectrogram

# --- Filter + Whiten helpers ---
def bandpass(data, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def whiten(data):
    return (data - np.mean(data)) / np.std(data)

# --- Load real LIGO data around GW150914 ---
gps = 1126259460
duration = 4
sample_rate = 16384

strain = TimeSeries.fetch_open_data('H1', gps, gps + duration).resample(sample_rate)
strain_data = strain.value

# --- Generate GW150914 waveform (true signal) ---
hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                        mass1=36, mass2=29,
                        delta_t=1/sample_rate,
                        f_lower=20,
                        distance=410)
chirp = hp.numpy()
chirp = chirp[:len(strain_data)]
chirp_padded = np.zeros_like(strain_data)
start = len(strain_data) // 2 - len(chirp) // 2
chirp_padded[start:start + len(chirp)] = chirp

# --- Create spectrogram from noisy LIGO strain ---
filtered = bandpass(strain_data, sample_rate, 30, 350)
whitened = whiten(filtered)
f, t_spec, Sxx = spectrogram(whitened, fs=sample_rate, nperseg=2048, noverlap=1024)
Sxx_dB = 10 * np.log10(Sxx + 1e-12)
Sxx_dB = np.clip(Sxx_dB, a_min=-100, a_max=None)

input_spec = torch.tensor(Sxx_dB[None, None, :, :], dtype=torch.float32)

# --- Define neural network architecture ---
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

# --- Load trained model ---
input_shape = input_spec.shape[2:]
output_length = len(chirp_padded[:Sxx.shape[1] * (2048 - 1024)])

model = SpectrogramToWaveformNet(input_shape, output_length)
model_path = "/mnt/c/Users/nickr/Downloads/MyGWData/denoising_cnn_model.pt"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# --- Predict denoised waveform ---
with torch.no_grad():
    prediction = model(input_spec).squeeze().numpy()

# --- Plot comparison ---
time = np.linspace(0, duration, len(prediction))
plt.figure(figsize=(12, 6))
plt.plot(time, prediction, label="🔵 NN Predicted Denoised Signal", linewidth=1.2)
plt.plot(time, chirp_padded[:len(prediction)], label="🟢 True GW150914 Waveform", alpha=0.8)
plt.plot(time, strain_data[:len(prediction)], label="🟣 Raw H1 Strain", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Denoising GW150914 with Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gw150914_denoising_comparison.png", dpi=300)
print("✅ Plot saved as gw150914_denoising_comparison.png")
