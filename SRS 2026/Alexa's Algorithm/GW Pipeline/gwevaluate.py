import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import log2
from scipy import signal as scipy_signal
from pycbc.filter import highpass
from pycbc.types import TimeSeries
from pycbc.psd import aLIGOZeroDetHighPower

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CACHE_PATH  = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/cache"
PLOTS_PATH  = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/plots"
SAMPLE_RATE = 4096
F_LOWER     = 20.0

os.makedirs(PLOTS_PATH, exist_ok=True)

# -------------------------------------------------------
# LOAD CACHES
# -------------------------------------------------------
def load_cache(name):
    path = os.path.join(CACHE_PATH, f"{name}.npz")
    data = np.load(path, allow_pickle=True)
    signals = [np.array(s, dtype=np.float32) for s in data["signals"]]
    names   = list(data["names"])
    print(f"Loaded {len(signals)} signals from {name}.npz")
    return signals, names

clean_data,    clean_names    = load_cache("clean_cache")
noisy_data,    noisy_names    = load_cache("noisy_cache")
denoised_data, denoised_names = load_cache("denoised_cache")

metadata = np.load(os.path.join(CACHE_PATH, "clean_metadata.npy"),
                   allow_pickle=True).item()


# -------------------------------------------------------
# METRIC 1: SNR
# -------------------------------------------------------
def compute_snr(clean, test):
    n = min(len(clean), len(test))
    clean, test    = clean[:n], test[:n]
    signal_power   = np.mean(clean ** 2)
    noise_power    = np.mean((clean - test) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# -------------------------------------------------------
# METRIC 2: MSE
# -------------------------------------------------------
def compute_mse(clean, test):
    n = min(len(clean), len(test))
    return np.mean((clean[:n] - test[:n]) ** 2)


# -------------------------------------------------------
# METRIC 3: MATCHED FILTER OVERLAP
# Measures how well the test signal correlates with the clean template
# Returns a value between 0 (no match) and 1 (perfect match)
# -------------------------------------------------------
def compute_overlap(clean, test, sample_rate=SAMPLE_RATE, f_lower=F_LOWER):
    from pycbc.filter import overlap_cplx

    n = min(len(clean), len(test))

    # Truncate to nearest power of 2 for FFT compatibility
    n = 2 ** int(np.log2(n))

    clean = clean[:n].astype(np.float64)
    test  = test[:n].astype(np.float64)

    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / (n * delta_t)
    flen    = n // 2 + 1

    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    h = TimeSeries(clean, delta_t=delta_t)
    s = TimeSeries(test,  delta_t=delta_t)

    # overlap_cplx with normalized=True already returns <h|s>/sqrt(<h|h><s|s>)
    # Result is complex — take abs to get magnitude regardless of phase difference
    try:
        ov = overlap_cplx(h, s, psd=psd, low_frequency_cutoff=f_lower, normalized=True)
        return float(abs(ov))
    except Exception as e:
        print(f"    Overlap calculation failed: {e}")
        return float('nan')


# -------------------------------------------------------
# SVD ENTROPY (for annotation)
# -------------------------------------------------------
SVD_EMBED_DIM = 20

def svd_entropy(signal, embed_dim=SVD_EMBED_DIM):
    signal = np.array(signal, dtype=np.float32)
    n = len(signal)
    if n < embed_dim * 2:
        return 1.0
    num_rows = n - embed_dim + 1
    matrix   = np.array([signal[i:i + embed_dim] for i in range(num_rows)])
    _, sv, _ = np.linalg.svd(matrix, full_matrices=False)
    total    = np.sum(sv)
    if total == 0:
        return 1.0
    w = sv / total
    w = w[w > 0]
    H = -np.sum(w * np.log2(w))
    maxH = log2(len(w))
    return H / maxH if maxH > 0 else 0.0


# -------------------------------------------------------
# SPECTROGRAM
# Uses scipy spectrogram with log frequency axis.
# nperseg=512 gives ~125ms time resolution at 4096 Hz,
# which is well matched to the chirp sweep rate.
# -------------------------------------------------------
def plot_spectrogram(ax, signal, sample_rate, title):
    f, t, Sxx = scipy_signal.spectrogram(
        signal.astype(np.float64),
        fs       = sample_rate,
        nperseg  = 512,
        noverlap = 480,
        scaling  = 'spectrum',
    )
    mask = f >= F_LOWER
    power_db = 10 * np.log10(Sxx[mask] + 1e-40)
    ax.pcolormesh(t, f[mask], power_db, shading='auto', cmap='inferno')
    ax.set_yscale('log')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(F_LOWER, sample_rate / 2)
    ax.set_title(title, fontsize=10)


# -------------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------------
N = len(denoised_data)

# Collect results grouped by parameter group
group_results = {}

col_w = 32
print("=" * 105)
print(f"{'Signal':<{col_w}} {'SVD N':>7} {'SVD D':>7} {'SNR N':>8} {'SNR D':>8} "
      f"{'ΔSNR':>7} {'MSE N':>9} {'MSE D':>9} {'Overlap':>8}")
print("=" * 105)

all_results = []

for i in range(N):
    name = denoised_names[i]

    # Match clean signal by name
    try:
        ci = clean_names.index(name)
        ni = noisy_names.index(name)
    except ValueError:
        print(f"  Warning: {name} not found in clean/noisy cache — skipping.")
        continue

    clean    = clean_data[ci]
    noisy    = noisy_data[ni]
    denoised = denoised_data[i]

    svd_n    = svd_entropy(noisy)
    svd_d    = svd_entropy(denoised)
    snr_n    = compute_snr(clean, noisy)
    snr_d    = compute_snr(clean, denoised)
    mse_n    = compute_mse(clean, noisy)
    mse_d    = compute_mse(clean, denoised)
    overlap  = compute_overlap(clean, denoised)
    delta    = snr_d - snr_n

    group = metadata.get(name, {}).get("group", "unknown")
    if group not in group_results:
        group_results[group] = []
    group_results[group].append({
        "name": name, "svd_n": svd_n, "svd_d": svd_d,
        "snr_n": snr_n, "snr_d": snr_d, "delta": delta,
        "mse_n": mse_n, "mse_d": mse_d, "overlap": overlap,
        "clean": clean, "noisy": noisy, "denoised": denoised,
    })
    all_results.append(group_results[group][-1])

    label = name[:col_w - 2]
    print(f"{label:<{col_w}} {svd_n:>7.4f} {svd_d:>7.4f} {snr_n:>8.2f} {snr_d:>8.2f} "
          f"{delta:>+7.2f} {mse_n:>9.6f} {mse_d:>9.6f} {overlap:>8.4f}")

print("=" * 105)

# Group averages
print("\nGroup averages:")
print("-" * 105)
for group, results in sorted(group_results.items()):
    avg_snr_n   = np.mean([r["snr_n"]   for r in results])
    avg_snr_d   = np.mean([r["snr_d"]   for r in results])
    avg_delta   = np.mean([r["delta"]   for r in results])
    avg_mse_n   = np.mean([r["mse_n"]   for r in results])
    avg_mse_d   = np.mean([r["mse_d"]   for r in results])
    avg_overlap = np.mean([r["overlap"] for r in results])
    avg_svd_n   = np.mean([r["svd_n"]   for r in results])
    avg_svd_d   = np.mean([r["svd_d"]   for r in results])
    label = f"AVG {group}"
    print(f"{label:<{col_w}} {avg_svd_n:>7.4f} {avg_svd_d:>7.4f} {avg_snr_n:>8.2f} "
          f"{avg_snr_d:>8.2f} {avg_delta:>+7.2f} {avg_mse_n:>9.6f} {avg_mse_d:>9.6f} "
          f"{avg_overlap:>8.4f}")

print("\nHow to read these results:")
print("  SVD N → SVD D  : Should drop (lower entropy = less noisy structure)")
print("  SNR N → SNR D  : Should rise (higher dB = cleaner signal)")
print("  ΔSNR           : Positive = improvement")
print("  MSE N → MSE D  : Should drop (lower = closer to clean)")
print("  Overlap        : 0–1, higher = better chirp shape preserved (>0.97 = excellent)")


# -------------------------------------------------------
# PLOTS — ONE FIGURE PER SIGNAL
# -------------------------------------------------------
print("\nGenerating plots...")

for r in all_results:
    name     = r["name"]
    clean    = r["clean"]
    noisy    = r["noisy"]
    denoised = r["denoised"]

    n_samples = min(len(clean), len(noisy), len(denoised))
    time_axis = np.arange(n_samples) / SAMPLE_RATE

    clean    = clean[:n_samples]
    noisy    = noisy[:n_samples]
    denoised = denoised[:n_samples]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Signal: {name}", fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # --- LEFT COLUMN: waveforms ---
    ax_c = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[1, 0], sharex=ax_c)
    ax_d = fig.add_subplot(gs[2, 0], sharex=ax_c)

    ax_c.plot(time_axis, clean,    color='steelblue', linewidth=0.5)
    ax_c.set_title("Clean signal (reference)", fontsize=10)
    ax_c.set_ylabel("Strain")
    ax_c.grid(True, alpha=0.3)
    ax_c.annotate(f"SVD: {svd_entropy(clean):.4f}",
                  xy=(0.01, 0.88), xycoords='axes fraction', fontsize=8,
                  color='steelblue',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax_n.plot(time_axis, noisy,    color='crimson',   linewidth=0.5)
    ax_n.set_title("Noisy signal", fontsize=10)
    ax_n.set_ylabel("Strain")
    ax_n.grid(True, alpha=0.3)
    ax_n.annotate(f"SVD: {r['svd_n']:.4f}  |  SNR: {r['snr_n']:.2f} dB  |  MSE: {r['mse_n']:.6f}",
                  xy=(0.01, 0.88), xycoords='axes fraction', fontsize=8,
                  color='crimson',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax_d.plot(time_axis, denoised, color='seagreen',  linewidth=0.5)
    ax_d.set_title("Denoised signal", fontsize=10)
    ax_d.set_ylabel("Strain")
    ax_d.set_xlabel("Time (s)")
    ax_d.grid(True, alpha=0.3)
    ax_d.annotate(
        f"SVD: {r['svd_d']:.4f}  |  SNR: {r['snr_d']:.2f} dB  |  "
        f"MSE: {r['mse_d']:.6f}  |  ΔSNR: {r['delta']:+.2f} dB  |  Overlap: {r['overlap']:.4f}",
        xy=(0.01, 0.88), xycoords='axes fraction', fontsize=8,
        color='seagreen',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # --- RIGHT COLUMN: Q-transform spectrograms ---
    ax_qc = fig.add_subplot(gs[0, 1])
    ax_qn = fig.add_subplot(gs[1, 1])
    ax_qd = fig.add_subplot(gs[2, 1])

    try:
        plot_spectrogram(ax_qc, clean,    SAMPLE_RATE, "Spectrogram: clean")
        plot_spectrogram(ax_qn, noisy,    SAMPLE_RATE, "Spectrogram: noisy")
        plot_spectrogram(ax_qd, denoised, SAMPLE_RATE, "Spectrogram: denoised")
        ax_qd.set_xlabel("Time (s)")
    except Exception as e:
        ax_qc.text(0.5, 0.5, f"Spectrogram failed:\n{e}",
                   transform=ax_qc.transAxes, ha='center', va='center', fontsize=8)

    plot_file = os.path.join(PLOTS_PATH, f"plot_{name}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot_{name}.png")

print(f"\nAll plots saved to {PLOTS_PATH}")
