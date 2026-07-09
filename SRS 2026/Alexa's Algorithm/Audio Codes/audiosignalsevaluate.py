import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
from math import log2

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
NOISY_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"
DENOISED_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\denoised_output"
SAMPLE_RATE   = 22050
EMBED_DIM     = 5
TIME_DELAY    = 1

# -------------------------------------------------------
# LOAD FROM CACHE (fast)
# -------------------------------------------------------
CACHE_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\cache"

def load_from_cache(cache_name):
    cache_file = os.path.join(CACHE_PATH, f"{cache_name}.npz")
    data = np.load(cache_file, allow_pickle=True)
    signals = [np.array(s, dtype=np.float32) for s in data["signals"]]
    names   = list(data["names"])
    print(f"Loaded {len(signals)} signals from {cache_name}.npz")
    return signals, names

clean_data,    clean_names    = load_from_cache("clean_cache")
noisy_data,    noisy_names    = load_from_cache("noisy_cache")
denoised_data, denoised_names = load_from_cache("denoised_cache")


# -------------------------------------------------------
# --- SVD ENTROPY FUNCTION ---
SVD_EMBED_DIM = 40      # Number of rows in the trajectory matrix
                        # Higher = more sensitive to structure, slower to compute
                        # Recommended range: 20-50

def svd_entropy(signal, embed_dim=SVD_EMBED_DIM):

    signal = np.array(signal, dtype=np.float32)
    n = len(signal)

    # Need at least embed_dim * 2 samples to build a meaningful matrix
    if n < embed_dim * 2:
        return 1.0      # Too short to analyze — assume high entropy

    # Step 1: Build a trajectory matrix (Hankel matrix)
    # Each row is a sliding window of length embed_dim
    num_rows = n - embed_dim + 1
    matrix = np.array([signal[i:i + embed_dim] for i in range(num_rows)])

    # Step 2: Compute Singular Value Decomposition
    # singular_values tells us how much energy each component carries
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)

    # Step 3: Normalize singular values into a probability distribution
    total = np.sum(singular_values)
    if total == 0:
        return 1.0      # Flat signal — treat as maximum entropy
    weights = singular_values / total

    # Step 4: Compute Shannon entropy of the weight distribution
    weights = weights[weights > 0]      # Ignore zero weights to avoid log(0)
    entropy = -np.sum(weights * np.log2(weights))

    # Step 5: Normalize by maximum possible entropy
    max_entropy = log2(len(weights))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy

# -------------------------------------------------------
# EVALUATION METRIC 2: SIGNAL TO NOISE RATIO (SNR)
# Measures how much cleaner the denoised signal is vs clean
# Higher = better
# -------------------------------------------------------
def compute_snr(clean, denoised):
    # Match lengths in case of minor size differences
    min_len = min(len(clean), len(denoised))
    clean = clean[:min_len]
    denoised = denoised[:min_len]
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# -------------------------------------------------------
# EVALUATION METRIC 3: MEAN SQUARED ERROR (MSE)
# Measures average difference between denoised and clean
# Lower = better
# -------------------------------------------------------
def compute_mse(clean, denoised):
    min_len = min(len(clean), len(denoised))
    clean = clean[:min_len]
    denoised = denoised[:min_len]
    return np.mean((clean - denoised) ** 2)


# -------------------------------------------------------
# RUN EVALUATION ON FIRST 5 SIGNALS
# -------------------------------------------------------
print("=" * 60)
print(f"{'Signal':<35} {'SVD Before':>10} {'SVD After':>10} {'SNR (dB)':>9} {'MSE':>10}")
print("=" * 60)

snr_scores  = []
mse_scores  = []
svd_before_all = []
svd_after_all  = []

for i in range(min(5, len(denoised_data))):
    svd_before = svd_entropy(noisy_data[i])
    svd_after  = svd_entropy(denoised_data[i])
    snr       = compute_snr(clean_data[i], denoised_data[i])
    mse       = compute_mse(clean_data[i], denoised_data[i])

    snr_scores.append(snr)
    mse_scores.append(mse)
    svd_before_all.append(svd_before)
    svd_after_all.append(svd_after)

    # Trim filename for clean display
    name = denoised_names[i][:33]
    print(f"{name:<35} {svd_before:>9.4f} {svd_after:>9.4f} {snr:>9.2f} {mse:>10.6f}")

print("=" * 60)
print(f"{'AVERAGE':<35} {np.mean(pe_before_all):>9.4f} {np.mean(svd_after_all):>9.4f} {np.mean(snr_scores):>9.2f} {np.mean(mse_scores):>10.6f}")
print("=" * 60)

print("\nHow to read these results:")
print("  SVD Before → SVD After : Should drop (lower = less noisy)")
print("  SNR (dB)             : Higher is better (above 10 dB is good)")
print("  MSE                  : Lower is better (closer to 0 = closer to clean signal)")

# -------------------------------------------------------
# VISUALIZATION — ONE FIGURE PER SIGNAL
# -------------------------------------------------------
PLOTS_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

print("\nGenerating plots...")

for i in range(min(5, len(denoised_data))):

    # Build a shared time axis in seconds
    num_samples = min(len(clean_data[i]), len(noisy_data[i]), len(denoised_data[i]))
    time_axis = np.linspace(0, num_samples / SAMPLE_RATE, num=num_samples)

    # Trim all three signals to the same length
    clean    = clean_data[i][:num_samples]
    noisy    = noisy_data[i][:num_samples]
    denoised = denoised_data[i][:num_samples]

    # --- BUILD THE FIGURE ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Signal: {denoised_names[i]}", fontsize=13, fontweight='bold', y=1.01)

    # --- ROW 1: CLEAN SIGNAL ---
    axes[0].plot(time_axis, clean, color='steelblue', linewidth=0.6)
    axes[0].set_title("Clean Signal", fontsize=11)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate(f"SVD: {svd_entropy(clean):.4f}",
                     xy=(0.01, 0.88), xycoords='axes fraction',
                     fontsize=9, color='steelblue',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # --- ROW 2: NOISY SIGNAL ---
    axes[1].plot(time_axis, noisy, color='crimson', linewidth=0.6)
    axes[1].set_title("Noisy Signal", fontsize=11)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate(f"SVD: {svd_entropy(noisy):.4f}",
                     xy=(0.01, 0.88), xycoords='axes fraction',
                     fontsize=9, color='crimson',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # --- ROW 3: DENOISED SIGNAL ---
    axes[2].plot(time_axis, denoised, color='seagreen', linewidth=0.6)
    axes[2].set_title("Denoised Signal", fontsize=11)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].grid(True, alpha=0.3)
    axes[2].annotate(f"SVD: {svd_entropy(denoised):.4f}  |  SNR: {snr_scores[i]:.2f} dB  |  MSE: {mse_scores[i]:.6f}",
                     xy=(0.01, 0.88), xycoords='axes fraction',
                     fontsize=9, color='seagreen',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # --- SAVE TO DISK ---
    plot_filename = denoised_names[i].replace("denoised_", "plot_").replace(".wav", ".png")
    plot_path = os.path.join(PLOTS_PATH, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")

    # --- SHOW ON SCREEN ---
    plt.show()

print(f"\nAll plots saved to {PLOTS_PATH}")
