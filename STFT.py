import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.signal.windows import hann
from scipy.ndimage import median_filter
import time

def generate_chirp_signal(t, f0=35, f1=250, amp_func=None):
    """
    Generate a chirp signal with time-varying frequency and amplitude
    
    Parameters:
    t: time array
    f0: starting frequency (Hz)
    f1: ending frequency (Hz)
    amp_func: function for amplitude variation (default: exponential decay)
    """
    if amp_func is None:
        # Default: exponential amplitude growth (like GW chirp)
        amp_func = lambda t: 0.5 * (1 + 2 * t / t[-1])**2
    
    # Instantaneous frequency changes quadratically (like GW chirp)
    freq_inst = f0 + (f1 - f0) * (t / t[-1])**2
    
    # Phase is integral of frequency
    phase = 2 * np.pi * np.cumsum(freq_inst) * (t[1] - t[0])
    
    # Generate signal with time-varying amplitude
    amplitude = amp_func(t)
    signal = amplitude * np.sin(phase)
    
    return signal, freq_inst, amplitude

def add_uniform_noise(signal, noise_level=0.3):
    """Add uniform noise to signal"""
    noise = np.random.uniform(-noise_level, noise_level, len(signal))
    return signal + noise, noise

def stft_denoise(noisy_signal, fs, nperseg=512, noverlap=None, threshold_factor=0.2):
    """
    Denoise signal using STFT with magnitude thresholding
    Improved version for smoother output
    """
    if noverlap is None:
        noverlap = int(nperseg * 0.75)  # Higher overlap for smoother reconstruction
    
    # Use a smoother window
    window = 'hann'
    
    # Compute STFT
    f, t_stft, Zxx = stft(noisy_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    
    # Compute magnitude and phase
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Improved adaptive thresholding with smoothing
    threshold_map = np.zeros_like(magnitude)
    
    # Apply temporal smoothing to threshold estimates
    for i in range(len(f)):
        # Use running median for more stable thresholding
        window_size = min(21, magnitude.shape[1])  # Adaptive window size
        if window_size >= 3:
            # Apply median filter for smoother thresholding
            smoothed_mag = median_filter(magnitude[i, :], size=window_size)
            threshold_map[i, :] = smoothed_mag * (1 + threshold_factor)
        else:
            threshold_map[i, :] = np.median(magnitude[i, :]) * (1 + threshold_factor)
    
    # Smoother soft thresholding with gradual transition
    alpha = 0.95  # Slightly lower preservation factor for more denoising
    denoised_magnitude = np.where(
        magnitude > threshold_map,
        magnitude * alpha + (magnitude - threshold_map) * (1 - alpha),
        magnitude * 0.1  # Stronger suppression for bins below threshold
    )
    
    # Reconstruct complex spectrogram
    denoised_Zxx = denoised_magnitude * np.exp(1j * phase)
    
    # Inverse STFT with same parameters
    _, denoised_signal = istft(denoised_Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    
    return denoised_signal, f, t_stft, magnitude, denoised_magnitude

def calculate_snr(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean((noisy_signal - clean_signal)**2)
    return 10 * np.log10(signal_power / noise_power)

# Parameters
fs = 1000  # Sampling frequency (Hz)
duration = 2.0  # Duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate clean chirp signal (mimicking gravitational wave)
clean_signal, freq_inst, amplitude = generate_chirp_signal(t, f0=50, f1=300)

# Add uniform noise
noise_level = 1
noisy_signal, noise = add_uniform_noise(clean_signal, noise_level)

# Measure runtime
start_time = time.time()

denoised_signal, freq_bins, time_bins, original_magnitude, denoised_magnitude = stft_denoise(
    noisy_signal, fs, nperseg=512, noverlap=384, threshold_factor=0.2
)

runtime = time.time() - start_time

# Trim denoised signal to match original length
min_len = min(len(clean_signal), len(denoised_signal))
clean_signal = clean_signal[:min_len]
noisy_signal = noisy_signal[:min_len]
denoised_signal = denoised_signal[:min_len]
t = t[:min_len]

# Calculate SNR improvements
snr_original = calculate_snr(clean_signal, noisy_signal)
snr_denoised = calculate_snr(clean_signal, denoised_signal)

print(f"Original SNR: {snr_original:.2f} dB")
print(f"Denoised SNR: {snr_denoised:.2f} dB")
print(f"SNR improvement: {snr_denoised - snr_original:.2f} dB")
print(f"Runtime: {runtime:.3f} seconds")

# Create focused plot - zoomed in section
zoom_start = int(0.5 * fs)  # Start at 0.5 seconds
zoom_end = int(1.2 * fs)    # End at 1.2 seconds
t_zoom = t[zoom_start:zoom_end]
clean_zoom = clean_signal[zoom_start:zoom_end]
noisy_zoom = noisy_signal[zoom_start:zoom_end]
denoised_zoom = denoised_signal[zoom_start:zoom_end]

plt.figure(figsize=(12, 6))
plt.plot(t_zoom, clean_zoom, 'b-', label='Original Clean Signal', linewidth=2)
plt.plot(t_zoom, noisy_zoom, 'r-', alpha=0.6, label='Noisy Signal', linewidth=1)
plt.plot(t_zoom, denoised_zoom, 'g-', label='STFT Filtered Signal', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('STFT Denoising Results (Zoomed View)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
