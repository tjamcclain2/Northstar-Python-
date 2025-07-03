import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
from scipy.signal import stft, istft
import os

# ============================
# LOAD REAL STRAIN DATA
# ============================
input_path = r"C:\Users\kasim\Downloads\H-H1_GWOSC_16KHZ_R1-1268903496-32.txt"
strain = np.loadtxt(input_path)

# Get base name for output files
base = os.path.splitext(os.path.basename(input_path))[0]

fs = 16384  # Hz
duration = 32  # seconds
t = np.linspace(0, duration, len(strain), endpoint=False)

print(f"Loaded strain data: {len(strain)} samples at {fs} Hz over {duration} s")

# ============================
# DWT DENOISING FUNCTION
# ============================
def dwt_denoise(noisy_signal, wavelet='db6', levels=6, threshold_mode='soft'):
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=levels)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_detail = pywt.threshold(detail, threshold, mode=threshold_mode)
        denoised_coeffs.append(denoised_detail)
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(noisy_signal)]

# ============================
# DENOISE AND TIME IT
# ============================
total_start_time = time.time()
main_start_time = time.time()

denoised_strain = dwt_denoise(strain)

main_end_time = time.time()

# ============================
# STFT PARAMETERS
# ============================
nperseg = 4096
noverlap = nperseg // 2

frequencies, times_stft, Zxx = stft(
    denoised_strain, 
    fs=fs, 
    nperseg=nperseg, 
    noverlap=noverlap, 
    window='hann'
)

print(f"STFT shape: freqs={len(frequencies)}, times={len(times_stft)}")

# ============================
# PLOT: RAW VS DENOISED (FULL DURATION)
# ============================
plt.figure(figsize=(14, 4))
plt.plot(t, strain, label='Raw Strain', color='gray', alpha=0.7)
plt.plot(t, denoised_strain, label='DWT Denoised', color='blue', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Raw vs DWT Denoised Gravitational Wave Strain (Full 32 Seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================
# RECONSTRUCT BANDS (1 Hz bandwidth, 50-1000 Hz)
# ============================
bands = [(hz, hz+1) for hz in range(50, 1000)]

# Store all band signals in a list for combined output
band_signals = []
band_labels = []
band_max_amplitudes = []

# Do not plot the bands, just save them
for i, (low, high) in enumerate(bands):
    band_mask = (frequencies >= low) & (frequencies < high)
    Zxx_band = np.zeros_like(Zxx)
    Zxx_band[band_mask, :] = Zxx[band_mask, :]
    
    _, band_signal = istft(
        Zxx_band, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        window='hann'
    )
    band_signals.append(band_signal)
    band_labels.append(f"{low}-{high}Hz")
    band_max_amplitudes.append(np.max(np.abs(band_signal)))

# Only save bands with significant strain
max_band_amp = np.max(band_max_amplitudes)
threshold = max_band_amp * 0.01  # Save bands with >1% of max amplitude (adjust as needed)

# Filter bands
significant_signals = []
significant_labels = []
for label, signal, amp in zip(band_labels, band_signals, band_max_amplitudes):
    if amp > threshold:
        significant_signals.append(signal)
        significant_labels.append(label)

# Pad signals to the same length
maxlen = max(len(sig) for sig in significant_signals)
padded_signals = [np.pad(sig, (0, maxlen - len(sig)), constant_values=np.nan) for sig in significant_signals]

# Stack as columns and save as CSV
csv_data = np.column_stack(padded_signals)
header = ','.join(significant_labels)
box_folder = r"C:\Users\kasim\Box\DWT_LIGOdata"
csv_filename = os.path.join(box_folder, f"{base}_bands_significant.csv")

# Save as binary .npz (MUCH FASTER and smaller) without normalization
bin_filename = os.path.join(box_folder, f"{base}_bands_significant.npz")
bin_start_time = time.time()
np.savez(bin_filename, data=csv_data, labels=significant_labels)
bin_end_time = time.time()
print(f"Bands with significant strain saved to {bin_filename} (binary, fast)")
print(f"Binary file creation time: {bin_end_time - bin_start_time:.4f} seconds")

# Optionally, comment out the CSV save if you only want binary
# csv_start_time = time.time()
# np.savetxt(csv_filename, csv_data, delimiter=",", header=header, comments='')
# csv_end_time = time.time()
# print(f"Bands with significant strain saved to {csv_filename} (threshold: {threshold:.2e})")
# print(f"CSV file creation time: {csv_end_time - csv_start_time:.4f} seconds")

# ============================
# DIAGNOSTICS
# ============================
total_end_time = time.time()

print(f"DWT amplitude recovery: {np.max(np.abs(denoised_strain)) / np.max(np.abs(strain)):.3f}")

print(f"\nRuntime diagnostics:")
print(f"{'Main computation time (s):':<30} {main_end_time - main_start_time:.4f}")
print(f"{'Total script runtime (s):':<30} {total_end_time - total_start_time:.4f}")

# ============================
# SAVE OUTPUT
# ============================
denoised_filename = f"{base}_denoised.txt"
np.savetxt(denoised_filename, denoised_strain)
print(f"Denoised strain saved as: {denoised_filename}")
