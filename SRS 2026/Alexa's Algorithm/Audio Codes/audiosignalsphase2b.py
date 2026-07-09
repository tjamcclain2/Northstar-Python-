import os
import librosa
import numpy as np
from math import log2

# --- SETTINGS ---
NOISY_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"
CLEAN_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
SAMPLE_RATE = 22050

# -------------------------------------------------------
# LOAD FROM CACHE 
# -------------------------------------------------------
CACHE_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\cache"

def load_from_cache(cache_name):
    cache_file = os.path.join(CACHE_PATH, f"{cache_name}.npz")
    data = np.load(cache_file, allow_pickle=True)
    signals = [np.array(s, dtype=np.float32) for s in data["signals"]]
    names   = list(data["names"])
    print(f"Loaded {len(signals)} signals from {cache_name}.npz")
    return signals, names

clean_data, clean_names = load_from_cache("clean_cache")
noisy_data, noisy_names = load_from_cache("noisy_cache")

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
    num_rows = n - embed_dim + 1
    matrix = np.array([signal[i:i + embed_dim] for i in range(num_rows)])

    # Step 2: Compute Singular Value Decomposition
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)

    # Step 3: Normalize singular values into a probability distribution
    total = np.sum(singular_values)
    if total == 0:
        return 1.0      # Flat signal — treat as maximum entropy
    weights = singular_values / total

    # Step 4: Compute Shannon entropy of the weight distribution
    weights = weights[weights > 0]
    entropy = -np.sum(weights * np.log2(weights))

    # Step 5: Normalize by maximum possible entropy
    max_entropy = log2(len(weights))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy

# --- APPLY TO NOISY SIGNALS ---
entropy_values = []

for i, signal in enumerate(noisy_data[:5]):
    sv = svd_entropy(signal)
    entropy_values.append(sv)
    print(f"{noisy_names[i]} — SVD Entropy: {sv:.4f}")

print(f"\nAverage entropy across all signals: {np.mean(entropy_values):.4f}")

# --- SETTINGS ---
WINDOW_SIZE   = 1024        # Samples per window
HOP_SIZE      = 512         # Slide between windows (50% overlap)
NUM_EVOLUTIONS = 20         # Maximum iterative denoising passes
GAIN_FLOOR    = 0.05        # Minimum gain applied to highest-entropy windows
GAIN_STEEPNESS = 20.0       # Sigmoid steepness — higher = sharper transition around threshold
CONVERGENCE_EPS = 0.0005    # Stop early if avg entropy change falls below this
MAD_K         = 1.0         # Threshold = median + MAD_K * MAD  (increase to be less aggressive)


# --- STEP 1: COMPUTE ENTROPY MAP ACROSS THE SIGNAL ---
def compute_entropy_map(signal, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    entropy_map = []
    positions = []

    for start in range(0, len(signal) - window_size, hop_size):
        window = signal[start:start + window_size]
        sv = svd_entropy(window)
        entropy_map.append(sv)
        positions.append(start)

    return np.array(entropy_map), np.array(positions)


# --- STEP 2: THRESHOLD ESTIMATION ---
# Uses median + k * MAD, which is resistant to Laplacian heavy tails
def estimate_threshold(entropy_map, k=MAD_K):
    median = np.median(entropy_map)
    mad = np.median(np.abs(entropy_map - median))
    threshold = median + k * mad
    return float(np.clip(threshold, 0.0, 1.0))


# --- STEP 3: SMOOTH SIGMOID GAIN (replaces binary on/off) ---
# Maps entropy continuously: low entropy → gain near 1.0, high entropy → gain near GAIN_FLOOR
# The transition is smooth around the threshold, controlled by GAIN_STEEPNESS
def entropy_to_gain(entropy_map, threshold, floor=GAIN_FLOOR, steepness=GAIN_STEEPNESS):
    sigmoid = 1.0 / (1.0 + np.exp(steepness * (entropy_map - threshold)))
    # Rescale from [0, 1] to [floor, 1.0] so we never fully silence a window
    gain = floor + (1.0 - floor) * sigmoid
    return gain


# --- STEP 4: LOW-RANK RECONSTRUCTION WITH VECTORIZED DIAGONAL AVERAGING ---
def apply_gain_to_signal(signal, gain_map, positions, window_size=WINDOW_SIZE):
    output = np.array(signal, dtype=np.float32).copy()
    window_func = np.hanning(window_size)

    # Precompute anti-diagonal index arrays once (same for every window of this size)
    num_rows = SVD_EMBED_DIM
    num_cols = window_size - SVD_EMBED_DIM + 1
    row_idx = np.repeat(np.arange(num_rows), num_cols)
    col_idx = np.tile(np.arange(num_cols), num_rows)
    diag_idx = row_idx + col_idx      # Which output sample each matrix cell maps to

    for i, start in enumerate(positions):
        end = start + window_size
        if end > len(signal):
            break

        # With sigmoid gain, all windows get some processing — only skip if gain ≈ 1.0
        if gain_map[i] > 0.995:
            continue

        window = np.array(signal[start:end], dtype=np.float32)

        if num_cols < 1:
            continue

        # Step 1: Build trajectory matrix
        matrix = np.array([window[j:j + num_cols] for j in range(num_rows)])

        # Step 2: SVD decomposition
        U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Step 3: Determine rank from gain — lower gain = more aggressive reduction
        entropy_level = 1.0 - gain_map[i]
        energy_target = 1.0 - (entropy_level * 0.60)
        energy_target = np.clip(energy_target, 0.30, 0.98)

        total_energy = np.sum(singular_values ** 2)
        cumulative_energy = np.cumsum(singular_values ** 2)
        rank = np.searchsorted(cumulative_energy, energy_target * total_energy) + 1
        rank = max(1, min(rank, len(singular_values)))

        filtered_sv = np.zeros_like(singular_values)
        filtered_sv[:rank] = singular_values[:rank]

        # Step 4: Reconstruct from dominant components
        reconstructed = U @ np.diag(filtered_sv) @ Vt

        # Step 5: Vectorized anti-diagonal averaging (replaces double for-loop)
        recovered = np.zeros(window_size)
        counts    = np.zeros(window_size)
        np.add.at(recovered, diag_idx, reconstructed.ravel())
        np.add.at(counts,    diag_idx, 1)
        counts = np.where(counts > 0, counts, 1)
        recovered = recovered / counts

        # Step 6: Blend with original using hanning window for smooth edges
        blend = window_func / (window_func.max() + 1e-10)
        output[start:end] = (1 - blend) * signal[start:end] + blend * recovered

    return output


# --- STEP 5: FULL ENTROPY-REGULARIZED DENOISER WITH EARLY STOPPING ---
def entropy_regularized_denoise(signal, sample_rate,
                                 window_size=WINDOW_SIZE,
                                 hop_size=HOP_SIZE,
                                 num_evolutions=NUM_EVOLUTIONS):

    current_signal = np.array(signal, dtype=np.float32).copy()
    entropy_history = []

    # Compute adaptive threshold using median + MAD on the initial (noisy) entropy map
    init_map, _ = compute_entropy_map(current_signal)
    adaptive_threshold = estimate_threshold(init_map)
    print(f"  Adaptive threshold (median + {MAD_K}*MAD): {adaptive_threshold:.4f}")

    for evolution in range(num_evolutions):

        entropy_map, positions = compute_entropy_map(current_signal, window_size, hop_size)
        avg_entropy = float(np.mean(entropy_map))
        entropy_history.append(avg_entropy)
        print(f"  Evolution {evolution + 1}/{num_evolutions} — Avg Entropy: {avg_entropy:.4f}")

        # Early stopping: if improvement has stalled, no point continuing
        if evolution > 0 and abs(entropy_history[-2] - entropy_history[-1]) < CONVERGENCE_EPS:
            print(f"  Converged at evolution {evolution + 1} — stopping early.")
            break

        gain = entropy_to_gain(entropy_map, adaptive_threshold)
        current_signal = apply_gain_to_signal(current_signal, gain, positions, window_size)

    return current_signal, entropy_history


# --- APPLY TO SIGNALS ---
denoised_data = []
all_entropy_histories = []

for i, noisy_signal in enumerate(noisy_data[:5]):
    print(f"\nDenoising: {noisy_names[i]}")
    denoised_signal, entropy_history = entropy_regularized_denoise(noisy_signal, SAMPLE_RATE)
    denoised_data.append(denoised_signal)
    all_entropy_histories.append(entropy_history)

print(f"\nDone! {len(denoised_data)} signals denoised.")

import soundfile as sf

# --- SAVE DENOISED SIGNALS ---
DENOISED_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\denoised_output"

try:
    os.makedirs(DENOISED_PATH, exist_ok=True)
    print(f"\nSaving denoised signals to: {DENOISED_PATH}")

    for i, denoised_signal in enumerate(denoised_data):
        original_name = noisy_names[i].replace("noisy_", "denoised_")
        output_file = os.path.join(DENOISED_PATH, original_name)
        sf.write(output_file, denoised_signal, SAMPLE_RATE)
        print(f"Saved: {original_name}")

    print(f"\nDone! {len(denoised_data)} denoised files saved.")

except Exception as e:
    print(f"ERROR: Something went wrong — {e}")
