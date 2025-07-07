# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:27:18 2025

@author: nickr
"""

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from scipy.signal import butter, filtfilt, spectrogram
import numpy as np

# ------------------------------------------------------
# 1️⃣ Load both H1 + L1, same method as before
# ------------------------------------------------------

gps = event_gps("GW150914")
start = int(gps) - 4
end   = int(gps) + 4

Fs = 16384

strain_H1 = TimeSeries.fetch_open_data('H1', start, end, sample_rate=Fs)
strain_L1 = TimeSeries.fetch_open_data('L1', start, end, sample_rate=Fs)

# Bandpass (Butterworth)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(30, 350, Fs, order=4)
strain_H1_bp = filtfilt(b, a, strain_H1.value)
strain_L1_bp = filtfilt(b, a, strain_L1.value)

# ------------------------------------------------------
# 2️⃣ Run STFT for both
# ------------------------------------------------------

nperseg = 4096
noverlap = int(nperseg * 0.75)

f_H1, t_H1, Sxx_H1 = spectrogram(strain_H1_bp, fs=Fs, window='blackman',
                                 nperseg=nperseg, noverlap=noverlap)

f_L1, t_L1, Sxx_L1 = spectrogram(strain_L1_bp, fs=Fs, window='blackman',
                                 nperseg=nperseg, noverlap=noverlap)

# ------------------------------------------------------
# 3️⃣ Find index of time bin closest to merger
# ------------------------------------------------------

merger_offset = gps - (int(gps) - 4)  # relative to chunk start
print(f"Merger should occur ~{merger_offset:.3f} s in chunk")

idx_H1 = np.argmin(np.abs(t_H1 - merger_offset))
idx_L1 = np.argmin(np.abs(t_L1 - merger_offset))

print(f"H1 bin center at t = {t_H1[idx_H1]:.5f} s")
print(f"L1 bin center at t = {t_L1[idx_L1]:.5f} s")

# ------------------------------------------------------
# 4️⃣ Integrate power across all freqs at that bin
# ------------------------------------------------------

P_H1 = np.sum(Sxx_H1[:, idx_H1])
P_L1 = np.sum(Sxx_L1[:, idx_L1])

print(f"H1: Total power at merger bin = {P_H1:.3e}")
print(f"L1: Total power at merger bin = {P_L1:.3e}")

# ------------------------------------------------------
# 5️⃣ Compare
# ------------------------------------------------------

if P_H1 > P_L1:
    print("H1 sees stronger total power at merger.")
else:
    print("L1 sees stronger total power at merger.")
