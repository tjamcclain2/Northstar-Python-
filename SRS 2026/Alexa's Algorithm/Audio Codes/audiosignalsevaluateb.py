import matplotlib.pyplot as plt
import numpy as np
import os
from math import log2

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
NOISY_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"
DENOISED_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\denoised_output"
SAMPLE_RATE   = 22050

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
# SVD ENTROPY
# -------------------------------------------------------
SVD_EMBED_DIM = 40

def svd_entropy(signal, embed_dim=SVD_EMBED_DIM):
    signal = np.array(signal, dtype=np.float32)
    n = len(signal)
    if n < embed_dim * 2:
        return 1.0
    num_rows = n - embed_dim + 1
    matrix = np.array([signal[i:i + embed_dim] for i in range(num_rows)])
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    total = np.sum(singular_values)
    if total == 0:
        return 1.0
    weights = singular_values / total
    weights = weights[weights > 0]
    entropy = -np.sum(weights * np.log2(weights))
    max_entropy = log2(len(weights))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


# -------------------------------------------------------
# EVALUATION METRICS
# -------------------------------------------------------
def compute_snr(clean, test):
    min_len = min(len(clean), len(test))
    clean = clean[:min_len]
    test  = test[:min_len]
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((clean - test) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(clean, test):
    min_len = min(len(clean), len(test))
    clean = clean[:min_len]
    test  = test[:min_len]
    return np.mean((clean - test) ** 2)


# -------------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------------
N = min(5, len(denoised_data))

col_w = 30
print("=" * 95)
print(f"{'Signal':<{col_w}} {'SVD Noisy':>9} {'SVD Densd':>9} {'SNR Noisy':>10} {'SNR Densd':>10} {'ΔSNR':>7} {'MSE Noisy':>10} {'MSE Densd':>10}")
print("=" * 95)

svd_noisy_all    = []
svd_denoised_all = []
snr_noisy_all    = []
snr_denoised_all = []
mse_noisy_all    = []
mse_denoised_all = []

for i in range(N):
    svd_noisy    = svd_entropy(noisy_data[i])
    svd_denoised = svd_entropy(denoised_data[i])
    snr_noisy    = compute_snr(clean_data[i], noisy_data[i])
    snr_denoised = compute_snr(clean_data[i], denoised_data[i])
    mse_noisy    = compute_mse(clean_data[i], noisy_data[i])
    mse_denoised = compute_mse(clean_data[i], denoised_data[i])
    delta_snr    = snr_denoised - snr_noisy

    svd_noisy_all.append(svd_noisy)
    svd_denoised_all.append(svd_denoised)
    snr_noisy_all.append(snr_noisy)
    snr_denoised_all.append(snr_denoised)
    mse_noisy_all.append(mse_noisy)
    mse_denoised_all.append(mse_denoised)

    name = denoised_names[i][:col_w - 2]
    print(f"{name:<{col_w}} {svd_noisy:>9.4f} {svd_denoised:>9.4f} {snr_noisy:>10.2f} {snr_denoised:>10.2f} {delta_snr:>+7.2f} {mse_noisy:>10.6f} {mse_denoised:>10.6f}")

print("=" * 95)
avg_delta = np.mean(snr_denoised_all) - np.mean(snr_noisy_all)
print(f"{'AVERAGE':<{col_w}} {np.mean(svd_noisy_all):>9.4f} {np.mean(svd_denoised_all):>9.4f} "
      f"{np.mean(snr_noisy_all):>10.2f} {np.mean(snr_denoised_all):>10.2f} {avg_delta:>+7.2f} "
      f"{np.mean(mse_noisy_all):>10.6f} {np.mean(mse_denoised_all):>10.6f}")
print("=" * 95)

print("\nHow to read these results:")
print("  SVD Noisy → SVD Densd : Should drop (lower entropy = less noisy structure)")
print("  SNR Noisy → SNR Densd : Should rise (higher dB = cleaner signal)")
print("  ΔSNR                  : Positive = improvement, negative = the denoiser made things worse")
print("  MSE Noisy → MSE Densd : Should drop (lower = closer to clean signal)")


# -------------------------------------------------------
# VISUALIZATION — ONE FIGURE PER SIGNAL
# -------------------------------------------------------
PLOTS_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

print("\nGenerating plots...")

for i in range(N):

    num_samples = min(len(clean_data[i]), len(noisy_data[i]), len(denoised_data[i]))
    time_axis   = np.linspace(0, num_samples / SAMPLE_RATE, num=num_samples)

    clean    = clean_data[i][:num_samples]
    noisy    = noisy_data[i][:num_samples]
    denoised = denoised_data[i][:num_samples]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Signal: {denoised_names[i]}", fontsize=13, fontweight='bold', y=1.01)

    # --- ROW 1: CLEAN SIGNAL ---
    axes[0].plot(time_axis, clean, color='steelblue', linewidth=0.6)
    axes[0].set_title("Clean Signal (reference)", fontsize=11)
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
    axes[1].annotate(
        f"SVD: {svd_noisy_all[i]:.4f}  |  SNR: {snr_noisy_all[i]:.2f} dB  |  MSE: {mse_noisy_all[i]:.6f}",
        xy=(0.01, 0.88), xycoords='axes fraction',
        fontsize=9, color='crimson',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # --- ROW 3: DENOISED SIGNAL ---
    delta_snr = snr_denoised_all[i] - snr_noisy_all[i]
    axes[2].plot(time_axis, denoised, color='seagreen', linewidth=0.6)
    axes[2].set_title("Denoised Signal", fontsize=11)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].grid(True, alpha=0.3)
    axes[2].annotate(
        f"SVD: {svd_denoised_all[i]:.4f}  |  SNR: {snr_denoised_all[i]:.2f} dB  |  MSE: {mse_denoised_all[i]:.6f}  |  ΔSNR: {delta_snr:+.2f} dB",
        xy=(0.01, 0.88), xycoords='axes fraction',
        fontsize=9, color='seagreen',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    plot_filename = denoised_names[i].replace("denoised_", "plot_").replace(".wav", ".png")
    plot_path = os.path.join(PLOTS_PATH, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")

    plt.show()

print(f"\nAll plots saved to {PLOTS_PATH}")
