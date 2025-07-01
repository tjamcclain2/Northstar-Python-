import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
from scipy import signal

def generate_realistic_gw_chirp(t, f0=50, f1=300, duration=0.4, amplitude_scale=1.0):
    """
    Generate realistic GW chirp with proper amplitude scaling for testing
    """
    # Focus on the inspiral phase
    t_inspiral = t[t <= duration]
    
    if len(t_inspiral) == 0:
        t_inspiral = t[:len(t)//2]  # Use first half if duration too short
    
    # Time to coalescence (counts backwards)
    t_to_coal = duration - t_inspiral
    t_to_coal[t_to_coal <= 0] = 1e-6  # Avoid division by zero
    
    # Realistic frequency evolution: f(t) ∝ (t_to_coal)^(-3/8)
    freq_inst = f0 * (t_to_coal / duration)**(-3/8)
    freq_inst = np.clip(freq_inst, f0, f1)
    
    # Amplitude evolution: grows as f^(2/3) (energy conservation)
    amplitude_evolution = (freq_inst / f0)**(2/3)
    # Scale to reasonable amplitude for testing (not realistic 1e-21!)
    amplitude = amplitude_scale * 0.1 * amplitude_evolution
    
    # Phase evolution - cumulative integral of frequency
    dt = t[1] - t[0] if len(t) > 1 else 1/2048
    phase = 2 * np.pi * np.cumsum(freq_inst) * dt
    
    # Generate the chirp signal
    signal_inspiral = amplitude * np.cos(phase)
    
    # Create full signal array
    full_signal = np.zeros(len(t))
    full_signal[:len(signal_inspiral)] = signal_inspiral
    
    return full_signal

# Parameters
fs = 2048  # Hz (typical LIGO sampling rate)
duration = 1.0  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate realistic GW chirp with visible amplitude
clean_signal = generate_realistic_gw_chirp(t, amplitude_scale=5.0)

def add_realistic_noise(signal, snr_target=8):
    """
    Add noise with proper SNR scaling
    """
    # White noise
    noise = np.random.normal(0, 1, len(signal))
    
    # Calculate SNR based on signal region only
    signal_mask = np.abs(signal) > 0.01 * np.max(np.abs(signal))
    
    if np.any(signal_mask):
        signal_power = np.var(signal[signal_mask])
    else:
        signal_power = np.var(signal)
    
    noise_power = np.var(noise)
    
    # Scale noise to achieve target SNR
    if noise_power > 0:
        noise_scale = np.sqrt(signal_power / (noise_power * snr_target**2))
        scaled_noise = noise * noise_scale
    else:
        scaled_noise = noise
    
    return signal + scaled_noise, scaled_noise

def dwt_denoise(noisy_signal, wavelet='db6', levels=6, threshold_mode='soft'):
    """
    Simplified DWT denoising
    """
    try:
        # Decompose
        coeffs = pywt.wavedec(noisy_signal, wavelet, level=levels)
        
        # Estimate noise level from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
        
        # Apply thresholding to detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation
        for detail in coeffs[1:]:
            denoised_detail = pywt.threshold(detail, threshold, mode=threshold_mode)
            denoised_coeffs.append(denoised_detail)
        
        # Reconstruct
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        
        # Ensure same length
        if len(denoised_signal) != len(noisy_signal):
            denoised_signal = denoised_signal[:len(noisy_signal)]
        
        return denoised_signal
        
    except Exception as e:
        print(f"Error in DWT denoising: {e}")
        return noisy_signal

def calculate_snr(clean_signal, noisy_signal):
    """Calculate SNR focusing on signal region"""
    # Find signal region (non-zero parts)
    signal_mask = np.abs(clean_signal) > 0.01 * np.max(np.abs(clean_signal))
    
    if np.any(signal_mask):
        signal_power = np.mean(clean_signal[signal_mask]**2)
        noise_power = np.mean((noisy_signal[signal_mask] - clean_signal[signal_mask])**2)
    else:
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean((noisy_signal - clean_signal)**2)
    
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_match(signal1, signal2):
    """Calculate match (normalized correlation) between two signals"""
    # Normalize signals
    s1_norm = signal1 / np.sqrt(np.sum(signal1**2))
    s2_norm = signal2 / np.sqrt(np.sum(signal2**2))
    
    # Calculate match
    match = np.abs(np.sum(s1_norm * s2_norm))
    return match

# Main analysis - simplified and faster
print("GW Chirp Signal Analysis with DWT Denoising")
print("=" * 50)

total_start_time = time.time()

# Test different SNR levels - reduced for speed
snr_levels = [5, 10, 15]
results = {}

print("\nTesting denoising at different SNR levels...")
main_start_time = time.time()

for snr in snr_levels:
    print(f"Processing SNR = {snr}...")
    
    # Add realistic noise
    noisy_signal, noise = add_realistic_noise(clean_signal, snr_target=snr)
    original_snr = calculate_snr(clean_signal, noisy_signal)
    
    # Apply denoising with optimal parameters
    denoised = dwt_denoise(noisy_signal, wavelet='db6', levels=6)
    
    # Calculate metrics
    snr_denoised = calculate_snr(clean_signal, denoised)
    match = calculate_match(clean_signal, denoised)
    improvement = snr_denoised - original_snr
    
    results[snr] = {
        'original_snr': original_snr,
        'denoised_snr': snr_denoised,
        'improvement': improvement,
        'match': match,
        'noisy': noisy_signal,
        'denoised': denoised
    }
    
    print(f"  Original SNR: {original_snr:.2f} dB")
    print(f"  Denoised SNR: {snr_denoised:.2f} dB")
    print(f"  Improvement: {improvement:.2f} dB")
    print(f"  Match: {match:.3f}")

main_end_time = time.time()

# Create visualizations
print("\nGenerating plots...")

# Plot 1: Clean signal
plt.figure(figsize=(12, 4))
plt.plot(t, clean_signal, 'b-', linewidth=2, label='Clean GW Chirp')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Clean Gravitational Wave Chirp Signal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Comparison for each SNR
fig, axes = plt.subplots(len(snr_levels), 1, figsize=(14, 4*len(snr_levels)))
if len(snr_levels) == 1:
    axes = [axes]

for i, snr in enumerate(snr_levels):
    ax = axes[i]
    result = results[snr]
    
    # Plot signals
    ax.plot(t, clean_signal, 'b-', label='Clean GW', linewidth=2)
    ax.plot(t, result['noisy'], 'gray', alpha=0.7, label='Noisy', linewidth=1)
    ax.plot(t, result['denoised'], 'r-', label='DWT Denoised', linewidth=2)
    
    ax.set_title(f'SNR {snr} dB → {result["denoised_snr"]:.1f} dB '
                f'(+{result["improvement"]:.1f} dB, Match: {result["match"]:.3f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits to show signal clearly
    max_amp = np.max(np.abs(clean_signal)) * 1.5
    ax.set_ylim(-max_amp, max_amp)

plt.tight_layout()
plt.show()

# Plot 3: Performance summary
plt.figure(figsize=(10, 6))

# SNR improvement plot
plt.subplot(1, 2, 1)
original_snrs = [results[snr]['original_snr'] for snr in snr_levels]
denoised_snrs = [results[snr]['denoised_snr'] for snr in snr_levels]
improvements = [results[snr]['improvement'] for snr in snr_levels]

plt.plot(snr_levels, original_snrs, 'o-', label='Original SNR', linewidth=2, markersize=8)
plt.plot(snr_levels, denoised_snrs, 's-', label='Denoised SNR', linewidth=2, markersize=8)
plt.xlabel('Input SNR (dB)')
plt.ylabel('SNR (dB)')
plt.title('SNR Improvement')
plt.legend()
plt.grid(True, alpha=0.3)

# Match plot
plt.subplot(1, 2, 2)
matches = [results[snr]['match'] for snr in snr_levels]
plt.plot(snr_levels, matches, 'g^-', label='Match', linewidth=2, markersize=8)
plt.xlabel('Input SNR (dB)')
plt.ylabel('Match')
plt.title('Signal Fidelity (Match)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Summary table
print(f"\nFinal Summary:")
print("=" * 60)
print(f"{'SNR':<6} {'Original':<10} {'Denoised':<10} {'Improve':<8} {'Match':<8}")
print("-" * 60)

for snr in snr_levels:
    result = results[snr]
    print(f"{snr:<6} {result['original_snr']:<10.2f} {result['denoised_snr']:<10.2f} "
          f"{result['improvement']:<8.2f} {result['match']:<8.3f}")

# --- Runtime diagnostics ---
total_end_time = time.time()

print(f"\nRuntime diagnostics:")
print(f"{'Main computation time (s):':<30} {main_end_time - main_start_time:.4f}")
print(f"{'Total script runtime (s):':<30} {total_end_time - total_start_time:.4f}")

print(f"\nKey Features:")
print("- Realistic GW chirp signal generation")
print("- Optimized DWT denoising with db6 wavelet")
print("- Fast computation with reduced parameter search")
print("- Clear visualization of results")
print("- Match metric for signal fidelity assessment")