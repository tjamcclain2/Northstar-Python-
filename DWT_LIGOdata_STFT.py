import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
from scipy.signal import stft, istft

# ============================
# LOAD REAL STRAIN DATA
# ============================
strain = np.loadtxt(r"C:\\Users\\kasim\\Downloads\\L-L1_GWOSC_16KHZ_R1-1369419303-32.txt")

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
# RECONSTRUCT BANDS
# ============================
bands = [(20,40), (40, 60), (60,80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 180), (180, 200)]

fig, axes = plt.subplots(len(bands), 1, figsize=(12, 3 * len(bands)))

if len(bands) == 1:
    axes = [axes]

for i, (low, high) in enumerate(bands):
    band_mask = (frequencies >= low) & (frequencies <= high)
    Zxx_band = np.zeros_like(Zxx)
    Zxx_band[band_mask, :] = Zxx[band_mask, :]
    
    _, band_signal = istft(
        Zxx_band, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        window='hann'
    )
    
    ax = axes[i]
    ax.plot(t[:len(band_signal)], band_signal, label=f'{low}-{high} Hz Band')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title(f"Reconstructed Time Slice: {low}-{high} Hz Band")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
np.savetxt("L-L1_GWOSC_16KHZ_R1-1369419303-32_denoised.txt", denoised_strain)
print("Denoised strain saved as: L-L1_GWOSC_16KHZ_R1-1369419303-32_denoised.txt")
