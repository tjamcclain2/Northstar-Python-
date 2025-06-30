import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.signal.windows import hann
from scipy.ndimage import median_filter
import time

# Start total runtime timer
script_start_time = time.time()

def generate_chirp_signal(t, f0=35, f1=250, amp_func=None, jitter_level=0.02):
    """Generate a discrete chirp signal mimicking GW detector readings."""
    if amp_func is None:
        amp_func = lambda t: 0.5 * (1 + 2 * t / t[-1])**2

    freq_inst = np.linspace(f0, f1, len(t))
    amplitude = amp_func(t)
    amplitude *= 1 + np.random.normal(0, jitter_level, size=len(t))
    amplitude = np.clip(amplitude, 0, None)
    phase = 2 * np.pi * np.cumsum(freq_inst) * (t[1] - t[0])
    phase += np.random.normal(0, jitter_level * 2 * np.pi, size=len(t))
    signal = amplitude * np.sin(phase)
    return signal, freq_inst, amplitude

def add_uniform_noise(signal, noise_level=0.3):
    """Add uniform noise to signal"""
    noise = np.random.uniform(-noise_level, noise_level, len(signal))
    return signal + noise, noise

def debug_stft_reconstruction(signal, fs, nperseg=512, noverlap=None):
    """Test if STFT->ISTFT reconstruction is perfect"""
    if noverlap is None:
        noverlap = int(nperseg * 0.75)
    
    # Forward and backward transform
    f, t_stft, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    _, reconstructed = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    
    # Compare lengths and trim if needed
    min_len = min(len(signal), len(reconstructed))
    original_trimmed = signal[:min_len]
    reconstructed_trimmed = reconstructed[:min_len]
    
    # Calculate reconstruction error
    mse = np.mean((original_trimmed - reconstructed_trimmed)**2)
    max_error = np.max(np.abs(original_trimmed - reconstructed_trimmed))
    
    print(f"STFT Reconstruction Test:")
    print(f"  MSE: {mse:.2e}")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Perfect reconstruction: {mse < 1e-10}")
    
    return mse < 1e-10

def simple_spectral_subtraction(noisy_signal, fs, nperseg=512, noverlap=None, 
                               alpha=1.0, beta=0.1, debug=True):
    """
    Simple, conservative spectral subtraction with debugging
    """
    if noverlap is None:
        noverlap = int(nperseg * 0.75)
    
    # Compute STFT
    f, t_stft, Zxx = stft(noisy_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    if debug:
        print(f"STFT shape: {Zxx.shape}")
        print(f"Frequency bins: {len(f)}")
        print(f"Time frames: {len(t_stft)}")
    
    # Conservative noise estimation - use minimum values across time for each frequency
    noise_spectrum = np.percentile(magnitude, 10, axis=1, keepdims=True)  # 10th percentile
    
    if debug:
        print(f"Noise spectrum shape: {noise_spectrum.shape}")
        print(f"Noise spectrum range: {np.min(noise_spectrum):.3f} to {np.max(noise_spectrum):.3f}")
        print(f"Signal spectrum range: {np.min(magnitude):.3f} to {np.max(magnitude):.3f}")
    
    # Very conservative spectral subtraction
    subtracted_magnitude = magnitude - alpha * noise_spectrum
    
    # Apply high spectral floor to prevent over-subtraction
    spectral_floor = beta * magnitude
    final_magnitude = np.maximum(subtracted_magnitude, spectral_floor)
    
    if debug:
        # Check how much we're actually subtracting
        reduction_ratio = final_magnitude / magnitude
        print(f"Magnitude reduction ratio: {np.min(reduction_ratio):.3f} to {np.max(reduction_ratio):.3f}")
        print(f"Average reduction: {np.mean(reduction_ratio):.3f}")
        
        # Count how many bins are at spectral floor
        at_floor = np.sum(final_magnitude == spectral_floor)
        total_bins = final_magnitude.size
        print(f"Bins at spectral floor: {at_floor}/{total_bins} ({100*at_floor/total_bins:.1f}%)")
    
    # Reconstruct
    denoised_Zxx = final_magnitude * np.exp(1j * phase)
    _, denoised_signal = istft(denoised_Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    
    return denoised_signal, f, t_stft, magnitude, final_magnitude

def calculate_snr(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean((noisy_signal - clean_signal)**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def diagnose_denoising_failure(clean, noisy, denoised, method_name):
    """Diagnose why denoising might be failing"""
    print(f"\n=== DIAGNOSIS FOR {method_name.upper()} ===")
    
    # Power analysis
    clean_power = np.mean(clean**2)
    noisy_power = np.mean(noisy**2)
    denoised_power = np.mean(denoised**2)
    noise_power = np.mean((noisy - clean)**2)
    
    print(f"Signal powers:")
    print(f"  Clean signal: {clean_power:.4f}")
    print(f"  Noisy signal: {noisy_power:.4f}")
    print(f"  Denoised signal: {denoised_power:.4f}")
    print(f"  Added noise: {noise_power:.4f}")
    
    # Error analysis
    denoising_error = np.mean((denoised - clean)**2)
    original_error = np.mean((noisy - clean)**2)
    
    print(f"Mean squared errors:")
    print(f"  Original (noisy vs clean): {original_error:.4f}")
    print(f"  After denoising (denoised vs clean): {denoising_error:.4f}")
    print(f"  Error ratio: {denoising_error/original_error:.3f}")
    
    if denoising_error > original_error:
        print(f"  ⚠️  PROBLEM: Denoising increased error by {((denoising_error/original_error - 1)*100):.1f}%")
    else:
        print(f"  ✅ Denoising reduced error by {((1 - denoising_error/original_error)*100):.1f}%")
    
    # Amplitude analysis
    clean_peak = np.max(np.abs(clean))
    denoised_peak = np.max(np.abs(denoised))
    amplitude_ratio = denoised_peak / clean_peak
    
    print(f"Amplitude analysis:")
    print(f"  Clean peak: {clean_peak:.4f}")
    print(f"  Denoised peak: {denoised_peak:.4f}")
    print(f"  Amplitude recovery: {amplitude_ratio:.3f}")
    
    return denoising_error < original_error

# Parameters
fs = 1000
duration = 2.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

print("=" * 60)
print("DEBUGGING SPECTRAL SUBTRACTION SNR ISSUES")
print("=" * 60)

# Generate signals
clean_signal, freq_inst, amplitude = generate_chirp_signal(t, f0=50, f1=300)
noise_level = 0.5  # Reduced noise level for testing
noisy_signal, noise = add_uniform_noise(clean_signal, noise_level)

# Test STFT reconstruction first
print("\n1. Testing STFT reconstruction:")
perfect_reconstruction = debug_stft_reconstruction(clean_signal, fs)

if not perfect_reconstruction:
    print("⚠️  STFT reconstruction has errors - this could be causing SNR issues!")

# Calculate baseline SNR
original_snr = calculate_snr(clean_signal, noisy_signal)
print(f"\nOriginal SNR: {original_snr:.2f} dB")

# Test different parameters systematically
test_params = [
    {'alpha': 0.5, 'beta': 0.2, 'name': 'very_conservative'},
    {'alpha': 1.0, 'beta': 0.1, 'name': 'conservative'},
    {'alpha': 1.5, 'beta': 0.05, 'name': 'moderate'},
    {'alpha': 2.0, 'beta': 0.01, 'name': 'aggressive'},
]

# Measure execution time for the main computation (excluding plotting)
start_time = time.time()

# Main computation: denoising loop
best_snr = -float('inf')
best_method = None
results = {}
for params in test_params:
    print(f"\n" + "="*50)
    print(f"Testing {params['name']} parameters:")
    print(f"Alpha: {params['alpha']}, Beta: {params['beta']}")
    
    # Run denoising
    denoised_signal, f, t_stft, orig_mag, final_mag = simple_spectral_subtraction(
        noisy_signal, fs, alpha=params['alpha'], beta=params['beta'], debug=True
    )
    
    # Trim to same length
    min_len = min(len(clean_signal), len(denoised_signal))
    clean_trimmed = clean_signal[:min_len]
    noisy_trimmed = noisy_signal[:min_len]
    denoised_trimmed = denoised_signal[:min_len]
    
    # Calculate SNR
    denoised_snr = calculate_snr(clean_trimmed, denoised_trimmed)
    snr_improvement = denoised_snr - original_snr
    
    print(f"Results:")
    print(f"  Denoised SNR: {denoised_snr:.2f} dB")
    print(f"  SNR improvement: {snr_improvement:.2f} dB")
    
    # Diagnose if this method failed
    success = diagnose_denoising_failure(clean_trimmed, noisy_trimmed, denoised_trimmed, params['name'])
    
    results[params['name']] = {
        'signal': denoised_trimmed,
        'snr': denoised_snr,
        'improvement': snr_improvement,
        'success': success
    }
    
    if snr_improvement > best_snr:
        best_snr = snr_improvement
        best_method = params['name']

end_time = time.time()
exec_runtime = end_time - start_time

print(f"\n" + "="*60)
print("FINAL RESULTS SUMMARY:")
print("="*60)
print(f"Original SNR: {original_snr:.2f} dB")
print(f"Best method: {best_method}")
print(f"Best SNR improvement: {best_snr:.2f} dB")

# Show results for all methods
print(f"\nAll methods comparison:")
print(f"{'Method':<15} {'SNR (dB)':<10} {'Improvement':<12} {'Success':<8}")
print("-" * 50)
for method_name, result in results.items():
    success_mark = "✅" if result['success'] else "❌"
    print(f"{method_name:<15} {result['snr']:<10.2f} {result['improvement']:<12.2f} {success_mark}")

# Plot the best result
if best_method:
    zoom_start = int(0.5 * fs)
    zoom_end = int(1.2 * fs)
    t_zoom = t[zoom_start:zoom_end]
    clean_zoom = clean_signal[zoom_start:zoom_end]
    noisy_zoom = noisy_signal[zoom_start:zoom_end]
    best_denoised_zoom = results[best_method]['signal'][zoom_start:zoom_end]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_zoom, clean_zoom, 'b-', label='Clean Signal', linewidth=2)
    plt.plot(t_zoom, noisy_zoom, 'r-', alpha=0.6, label='Noisy Signal', linewidth=1)
    plt.plot(t_zoom, best_denoised_zoom, 'g-', label=f'Best Denoised ({best_method})', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Best Denoising Result: {best_method.title()}\nSNR Improvement: {best_snr:.2f} dB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Show power spectral density
    from scipy.signal import welch
    f_psd, clean_psd = welch(clean_signal, fs, nperseg=512)
    f_psd, noisy_psd = welch(noisy_signal, fs, nperseg=512)
    f_psd, denoised_psd = welch(results[best_method]['signal'], fs, nperseg=512)
    
    plt.semilogy(f_psd, clean_psd, 'b-', label='Clean', linewidth=2)
    plt.semilogy(f_psd, noisy_psd, 'r-', alpha=0.7, label='Noisy', linewidth=1)
    plt.semilogy(f_psd, denoised_psd, 'g-', label='Denoised', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print(f"Execution time (main computation only): {exec_runtime:.3f} seconds")
# Print total runtime (including plotting)
script_end_time = time.time()
total_runtime = script_end_time - script_start_time
print(f"Total script runtime (including plotting): {total_runtime:.3f} seconds")