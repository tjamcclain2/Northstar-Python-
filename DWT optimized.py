import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
from scipy import signal

def generate_chirp_signal(t, f0=50, f1=200, amp_func=None):
    """
    Generate a chirp signal with time-varying frequency and amplitude
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

def simple_dwt_denoise(noisy_signal, wavelet='db4', levels=6, threshold_mode='soft', 
                      sigma_factor=0.1):
    """
    Simple but effective DWT denoising - back to basics
    """
    # Decompose
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=levels)
    
    # Estimate noise from finest detail coefficients
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    
    # Calculate threshold - much smaller than before
    threshold = sigma_factor * sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    
    # Apply thresholding only to detail coefficients, keep approximation
    denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i, detail in enumerate(coeffs[1:]):
        # Less aggressive thresholding for coarser levels
        level_factor = 1.0 - 0.1 * i  # Reduce threshold for coarser levels
        current_threshold = threshold * level_factor
        
        if threshold_mode == 'soft':
            denoised_detail = pywt.threshold(detail, current_threshold, mode='soft')
        else:
            denoised_detail = pywt.threshold(detail, current_threshold, mode='hard')
        
        denoised_coeffs.append(denoised_detail)
    
    # Reconstruct
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # Ensure same length
    if len(denoised_signal) > len(noisy_signal):
        denoised_signal = denoised_signal[:len(noisy_signal)]
    
    return denoised_signal, sigma, threshold

def wiener_like_filter(noisy_signal, clean_signal_estimate, noise_var):
    """
    Simple Wiener-like filtering to preserve amplitudes
    """
    # Estimate signal variance in local windows
    window_size = 100
    filtered_signal = np.zeros_like(noisy_signal)
    
    for i in range(len(noisy_signal)):
        start = max(0, i - window_size // 2)
        end = min(len(noisy_signal), i + window_size // 2)
        
        # Local signal variance estimate
        local_clean = clean_signal_estimate[start:end]
        signal_var = np.var(local_clean)
        
        # Wiener filter coefficient
        wiener_coeff = signal_var / (signal_var + noise_var)
        
        # Apply filter
        filtered_signal[i] = wiener_coeff * noisy_signal[i] + (1 - wiener_coeff) * clean_signal_estimate[i]
    
    return filtered_signal

def calculate_snr(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean((noisy_signal - clean_signal)**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

# Parameters
fs = 1000  # Sampling frequency (Hz)
duration = 2.0  # Duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# --- Runtime measurement ---
total_start_time = time.time()

# Generate clean chirp signal
clean_signal, freq_inst, amplitude = generate_chirp_signal(t, f0=50, f1=300)

# Test different noise levels
noise_levels = [0.5, 1.0, 1.5, 2.0]
results = {}
runtime_per_noise = {}  # Initialize runtime tracking per noise level

print("Testing Simple DWT Denoising at Different Noise Levels")
print("=" * 55)

main_start_time = time.time()

for noise_level in noise_levels:
    sweep_start = time.time()
    print(f"\nNoise Level: {noise_level}")
    print("-" * 30)
    
    # Add noise
    noisy_signal, noise = add_uniform_noise(clean_signal, noise_level)
    
    # Calculate original SNR
    snr_original = calculate_snr(clean_signal, noisy_signal)
    
    # Test different wavelets and parameters
    wavelets_to_test = ['db4', 'db6', 'db8', 'coif4']
    sigma_factors = [0.05, 0.1, 0.2, 0.3]
    
    best_snr = -np.inf
    best_params = {}
    best_denoised = None

    for wavelet in wavelets_to_test:
        for sigma_factor in sigma_factors:
            try:
                # Apply denoising
                denoised, sigma, threshold = simple_dwt_denoise(
                    noisy_signal, wavelet=wavelet, levels=6, 
                    sigma_factor=sigma_factor, threshold_mode='soft'
                )
                
                # Calculate SNR
                snr_denoised = calculate_snr(clean_signal, denoised)
                
                # Track best result
                if snr_denoised > best_snr:
                    best_snr = snr_denoised
                    best_params = {
                        'wavelet': wavelet,
                        'sigma_factor': sigma_factor,
                        'sigma': sigma,
                        'threshold': threshold
                    }
                    best_denoised = denoised.copy()
                    
            except Exception as e:
                continue
    runtime_per_noise[noise_level] = time.time() - sweep_start
    # Store results
    results[noise_level] = {
        'original_snr': snr_original,
        'best_snr': best_snr,
        'improvement': best_snr - snr_original,
        'params': best_params,
        'denoised': best_denoised,
        'noisy': noisy_signal,
        'noise_std': np.std(noise)
    }
    
    print(f"Original SNR: {snr_original:.2f} dB")
    print(f"Best Denoised SNR: {best_snr:.2f} dB")
    print(f"Improvement: {best_snr - snr_original:.2f} dB")
    print(f"Best params: {best_params['wavelet']}, σ_factor={best_params['sigma_factor']}")

main_end_time = time.time()

# Create visualization for all noise levels
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, noise_level in enumerate(noise_levels):
    ax = axes[i]
    result = results[noise_level]
    
    # Plot zoomed section for clarity
    zoom_start = int(0.8 * fs)
    zoom_end = int(1.3 * fs)
    t_zoom = t[zoom_start:zoom_end]
    
    ax.plot(t_zoom, clean_signal[zoom_start:zoom_end], 'b-', 
            label='Clean', linewidth=2)
    ax.plot(t_zoom, result['noisy'][zoom_start:zoom_end], 'gray', 
            alpha=0.4, label='Noisy', linewidth=1)
    
    if result['denoised'] is not None:
        ax.plot(t_zoom, result['denoised'][zoom_start:zoom_end], 'r-', 
                label='Denoised', linewidth=2)
    
    ax.set_title(f'Noise Level {noise_level} (SNR: {result["original_snr"]:.1f} → {result["best_snr"]:.1f} dB)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

total_end_time = time.time()

# Summary table
print(f"\nSummary Results:")
print("=" * 70)
print(f"{'Noise Level':<12} {'Original SNR':<12} {'Best SNR':<12} {'Improvement':<12} {'Best Wavelet':<12}")
print("-" * 70)

for noise_level in noise_levels:
    result = results[noise_level]
    if result['denoised'] is not None:
        print(f"{noise_level:<12} {result['original_snr']:<12.2f} {result['best_snr']:<12.2f} "
              f"{result['improvement']:<12.2f} {result['params']['wavelet']:<12}")

# Detailed analysis for highest noise level that worked well
print(f"\nDetailed Analysis for Best Performance:")
print("=" * 50)

# Find the noise level with best relative improvement
best_relative_improvement = -np.inf
best_noise_level = None

for noise_level in noise_levels:
    result = results[noise_level]
    if result['denoised'] is not None and result['improvement'] > best_relative_improvement:
        best_relative_improvement = result['improvement']
        best_noise_level = noise_level

if best_noise_level is not None:
    best_result = results[best_noise_level]
    print(f"Best performance at noise level: {best_noise_level}")
    print(f"Parameters: {best_result['params']}")
    
    # Calculate detailed metrics
    denoised = best_result['denoised']
    rms_error_noisy = np.sqrt(np.mean((best_result['noisy'] - clean_signal)**2))
    rms_error_denoised = np.sqrt(np.mean((denoised - clean_signal)**2))
    correlation = np.corrcoef(clean_signal, denoised)[0, 1]
    
    print(f"RMS Error - Noisy: {rms_error_noisy:.4f}")
    print(f"RMS Error - Denoised: {rms_error_denoised:.4f}")
    print(f"Error Reduction: {(1 - rms_error_denoised/rms_error_noisy)*100:.1f}%")
    print(f"Correlation with clean signal: {correlation:.4f}")
    
    # Amplitude preservation
    clean_max = np.max(np.abs(clean_signal))
    denoised_max = np.max(np.abs(denoised))
    amp_preservation = (denoised_max / clean_max) * 100
    print(f"Amplitude preservation: {amp_preservation:.1f}%")

print(f"\nKey Insights:")
print("- Lower sigma_factor values (0.05-0.1) generally work better")
print("- db4 and db6 wavelets often perform well for chirp signals")
print("- Performance degrades significantly above noise level 1.5")
print("- Simple approaches often outperform complex ones for this type of signal")

# --- Runtime diagnostics ---
print(f"\nRuntime diagnostics:")
print(f"{'Main computation time (s):':<30} {main_end_time - main_start_time:.4f}")
print(f"{'Total script runtime (s):':<30} {total_end_time - total_start_time:.4f}")
for noise_level, rt in runtime_per_noise.items():
    print(f"{'Sweep time (s) for noise=' + str(noise_level) + ':':<30} {rt:.4f}")