import numpy as np
import os
from math import log2
from pycbc.filter import highpass
from pycbc.types import TimeSeries

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CACHE_PATH    = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/cache"
DENOISED_PATH = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/denoised"
SAMPLE_RATE   = 4096

# SVD settings
SVD_EMBED_DIM   = 20        # Embedding dimension — smaller than audio version
                             # because windows are shorter at 4096 Hz

# Windowing settings — tuned for 4096 Hz and GW frequency band
WINDOW_SIZE     = 512        # ~125ms per window at 4096 Hz
HOP_SIZE        = 256        # 50% overlap

# Denoising behavior
NUM_EVOLUTIONS  = 100         # Maximum iterative passes
GAIN_FLOOR      = 0.05       # Minimum gain for highest-entropy windows
GAIN_STEEPNESS  = 20.0       # Sigmoid sharpness around threshold
CONVERGENCE_EPS      = 0.0005  # Early stopping threshold (per evolution)
CONVERGENCE_PATIENCE = 5       # Stop only after this many consecutive small-change evolutions
MAD_K_LOW       = 0.5        # Aggressive suppression for coherent (low-entropy) windows
MAD_K_HIGH      = 2.0        # Conservative suppression for incoherent (high-entropy) windows

# Highpass filter
F_LOWER         = 20.0       # Hz — remove sub-band content before denoising


# -------------------------------------------------------
# HIGHPASS FILTER
# -------------------------------------------------------
def apply_highpass(signal, sample_rate=SAMPLE_RATE, f_lower=F_LOWER):
    ts = TimeSeries(signal.astype(np.float64), delta_t=1.0/sample_rate)
    ts = highpass(ts, frequency=f_lower)
    return np.array(ts.data, dtype=np.float32)


# -------------------------------------------------------
# SVD ENTROPY
# -------------------------------------------------------
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
# ENTROPY MAP
# -------------------------------------------------------
def compute_entropy_map(signal, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    entropy_map = []
    positions   = []
    for start in range(0, len(signal) - window_size, hop_size):
        window = signal[start:start + window_size]
        entropy_map.append(svd_entropy(window))
        positions.append(start)
    return np.array(entropy_map), np.array(positions)


# -------------------------------------------------------
# THRESHOLD ESTIMATION (adaptive MAD_K based on local coherence)
# -------------------------------------------------------
# Each window gets its own MAD_K based on how coherent it is relative
# to the rest of the signal. Coherence is scored 0→1 using the entropy
# map: the lowest-entropy window scores 1.0 (most coherent, likely near
# merger), the highest-entropy window scores 0.0 (most disordered).
# MAD_K then interpolates between MAD_K_LOW (aggressive) and MAD_K_HIGH
# (conservative) — no prior knowledge of merger time required.
def estimate_threshold(entropy_map):
    median = np.median(entropy_map)
    mad    = np.median(np.abs(entropy_map - median))

    e_min = entropy_map.min()
    e_max = entropy_map.max()
    if e_max > e_min:
        coherence = (e_max - entropy_map) / (e_max - e_min)  # 1 = most coherent
    else:
        coherence = np.zeros_like(entropy_map)

    local_k   = MAD_K_HIGH - (MAD_K_HIGH - MAD_K_LOW) * coherence
    threshold = median + local_k * mad
    return np.clip(threshold, 0.0, 1.0)  # array, one value per window


# -------------------------------------------------------
# SIGMOID GAIN
# -------------------------------------------------------
def entropy_to_gain(entropy_map, threshold,
                    floor=GAIN_FLOOR, steepness=GAIN_STEEPNESS):
    sigmoid = 1.0 / (1.0 + np.exp(steepness * (entropy_map - threshold)))
    return floor + (1.0 - floor) * sigmoid


# -------------------------------------------------------
# LOW-RANK RECONSTRUCTION WITH VECTORIZED DIAGONAL AVERAGING
# -------------------------------------------------------
def apply_gain_to_signal(signal, gain_map, positions, window_size=WINDOW_SIZE):
    output      = np.array(signal, dtype=np.float32).copy()
    window_func = np.hanning(window_size)

    num_rows = SVD_EMBED_DIM
    num_cols = window_size - SVD_EMBED_DIM + 1
    row_idx  = np.repeat(np.arange(num_rows), num_cols)
    col_idx  = np.tile(np.arange(num_cols), num_rows)
    diag_idx = row_idx + col_idx

    for i, start in enumerate(positions):
        end = start + window_size
        if end > len(signal):
            break
        if gain_map[i] > 0.995:
            continue

        window = np.array(signal[start:end], dtype=np.float32)
        if num_cols < 1:
            continue

        matrix = np.array([window[j:j + num_cols] for j in range(num_rows)])
        U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)

        entropy_level = 1.0 - gain_map[i]
        energy_target = 1.0 - (entropy_level * 0.60)
        energy_target = np.clip(energy_target, 0.30, 0.98)

        total_energy      = np.sum(singular_values ** 2)
        cumulative_energy = np.cumsum(singular_values ** 2)
        rank = np.searchsorted(cumulative_energy, energy_target * total_energy) + 1
        rank = max(1, min(rank, len(singular_values)))

        filtered_sv        = np.zeros_like(singular_values)
        filtered_sv[:rank] = singular_values[:rank]
        reconstructed      = U @ np.diag(filtered_sv) @ Vt

        recovered = np.zeros(window_size)
        counts    = np.zeros(window_size)
        np.add.at(recovered, diag_idx, reconstructed.ravel())
        np.add.at(counts,    diag_idx, 1)
        counts    = np.where(counts > 0, counts, 1)
        recovered = recovered / counts

        blend = window_func / (window_func.max() + 1e-10)
        output[start:end] = (1 - blend) * signal[start:end] + blend * recovered

    return output


# -------------------------------------------------------
# FULL DENOISER
# -------------------------------------------------------
def entropy_regularized_denoise(signal, sample_rate=SAMPLE_RATE,
                                 window_size=WINDOW_SIZE,
                                 hop_size=HOP_SIZE,
                                 num_evolutions=NUM_EVOLUTIONS):

    # Highpass filter before denoising
    current_signal = apply_highpass(signal, sample_rate)
    entropy_history = []

    stall_count = 0
    for evolution in range(num_evolutions):
        entropy_map, positions = compute_entropy_map(current_signal, window_size, hop_size)
        avg_entropy            = float(np.mean(entropy_map))
        entropy_history.append(avg_entropy)
        print(f"  Evolution {evolution + 1}/{num_evolutions} — Avg Entropy: {avg_entropy:.4f}  "
              f"(threshold range: {estimate_threshold(entropy_map).min():.3f}–"
              f"{estimate_threshold(entropy_map).max():.3f})")

        if evolution > 0 and abs(entropy_history[-2] - entropy_history[-1]) < CONVERGENCE_EPS:
            stall_count += 1
            if stall_count >= CONVERGENCE_PATIENCE:
                print(f"  Converged at evolution {evolution + 1} "
                      f"({CONVERGENCE_PATIENCE} consecutive stalls) — stopping early.")
                break
        else:
            stall_count = 0

        threshold      = estimate_threshold(entropy_map)
        gain           = entropy_to_gain(entropy_map, threshold)
        current_signal = apply_gain_to_signal(current_signal, gain, positions, window_size)

    return current_signal, entropy_history


# -------------------------------------------------------
# LOAD CACHE AND DENOISE
# -------------------------------------------------------
os.makedirs(DENOISED_PATH, exist_ok=True)

cache_file = os.path.join(CACHE_PATH, "noisy_cache.npz")
data       = np.load(cache_file, allow_pickle=True)
signals    = [np.array(s, dtype=np.float32) for s in data["signals"]]
names      = list(data["names"])

print(f"Loaded {len(signals)} noisy signals from cache.")

# Set to None to process all signals
MAX_SIGNALS = 1
if MAX_SIGNALS is not None:
    signals = signals[:MAX_SIGNALS]
    names   = names[:MAX_SIGNALS]
    print(f"(Limited to {MAX_SIGNALS} signal(s) for testing)")

print(f"Saving denoised signals to: {DENOISED_PATH}\n")

denoised_metadata = {}

for i, noisy_signal in enumerate(signals):
    name = names[i]
    print(f"\nDenoising: {name}")
    denoised_signal, entropy_history = entropy_regularized_denoise(noisy_signal, SAMPLE_RATE)

    out_path = os.path.join(DENOISED_PATH, f"{name}.npy")
    np.save(out_path, denoised_signal)

    denoised_metadata[name] = {
        "entropy_history": entropy_history,
        "evolutions_run":  len(entropy_history),
        "final_entropy":   entropy_history[-1],
    }

    print(f"  Saved: {name}.npy")

np.save(os.path.join(DENOISED_PATH, "metadata.npy"), denoised_metadata)
print(f"\nDone! {len(signals)} signals denoised.")
