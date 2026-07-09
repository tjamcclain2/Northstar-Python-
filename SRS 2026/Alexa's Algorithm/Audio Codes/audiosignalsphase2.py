import os
import librosa
import numpy as np
from math import log2

# --- SETTINGS ---
NOISY_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"
CLEAN_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
SAMPLE_RATE = 22050

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

# --- APPLY TO NOISY SIGNALS ---
entropy_values = []

for i, signal in enumerate(noisy_data[:5]):
    sv = svd_entropy(signal)
    entropy_values.append(sv)
    print(f"{noisy_names[i]} — SVD Entropy: {sv:.4f}")

print(f"\nAverage entropy across all signals: {np.mean(entropy_values):.4f}")

# --- SETTINGS ---
WINDOW_SIZE = 1024          # Samples per window
HOP_SIZE = 512              # Slide between windows (50% overlap)
ENTROPY_THRESHOLD = 0.85     # Above this = high entropy = noise
NUM_EVOLUTIONS = 10          # Number of iterative denoising passes
ATTENUATION = 0.05           # How much to reduce high-entropy windows (0-1)


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


# --- STEP 2: CONVERT ENTROPY SCORES TO GAIN VALUES ---
# High entropy windows get reduced, low entropy windows are kept
def entropy_to_gain(entropy_map, threshold=ENTROPY_THRESHOLD, attenuation=ATTENUATION):
    gain = np.ones_like(entropy_map)                    # Start with everything at full volume
    high_entropy = entropy_map > threshold              # Find noisy windows
    gain[high_entropy] = attenuation                    # Turn them down
    return gain


from scipy.ndimage import uniform_filter1d

# --- LOW-RANK RECONSTRUCTION DENOISER ---
# Keeps dominant singular values, discards noise components
def apply_gain_to_signal(signal, gain_map, positions, window_size=WINDOW_SIZE):
    output = np.array(signal, dtype=np.float32).copy()
    weight = np.zeros(len(signal))
    window_func = np.hanning(window_size)

    for i, start in enumerate(positions):
        end = start + window_size
        if end > len(signal):
            break

        if gain_map[i] < 1.0:
            window = np.array(signal[start:end], dtype=np.float32)

            # Step 1: Build trajectory matrix from this window
            num_rows = SVD_EMBED_DIM
            num_cols = len(window) - SVD_EMBED_DIM + 1
            if num_cols < 1:
                continue
            matrix = np.array([window[j:j + num_cols] for j in range(num_rows)])

            # Step 2: SVD decomposition
            U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)

            # Step 3: Keep components that explain most of the energy
            # More aggressive for higher entropy windows
            entropy_level = 1.0 - gain_map[i]
            energy_target = 1.0 - (entropy_level * 0.60)   # High entropy = keep less energy
            energy_target = np.clip(energy_target, 0.30, 0.98)

            # Find minimum number of components to reach energy target
            total_energy = np.sum(singular_values ** 2)
            cumulative_energy = np.cumsum(singular_values ** 2)
            rank = np.searchsorted(cumulative_energy, energy_target * total_energy) + 1
            rank = max(1, min(rank, len(singular_values)))

            # Zero out weak singular values
            filtered_sv = np.zeros_like(singular_values)
            filtered_sv[:rank] = singular_values[:rank]

            # Step 4: Reconstruct matrix from dominant components only
            reconstructed = U @ np.diag(filtered_sv) @ Vt

            # Step 5: Average anti-diagonals to recover 1D signal (diagonal averaging)
            recovered = np.zeros(len(window))
            counts = np.zeros(len(window))
            for row in range(num_rows):
                for col in range(num_cols):
                    recovered[row + col] += reconstructed[row, col]
                    counts[row + col] += 1
            counts = np.where(counts > 0, counts, 1)
            recovered = recovered / counts

            # Step 6: Blend with original using hanning window for smooth edges
            blend = window_func / (window_func.max() + 1e-10)
            output[start:end] = (1 - blend) * signal[start:end] + blend * recovered

    return output

# --- STEP 4: FULL ENTROPY-REGULARIZED DENOISER ---
def entropy_regularized_denoise(signal, sample_rate,
                                 window_size=WINDOW_SIZE,
                                 hop_size=HOP_SIZE,
                                 threshold=ENTROPY_THRESHOLD,
                                 num_evolutions=NUM_EVOLUTIONS,
                                 attenuation=ATTENUATION):

    current_signal = np.array(signal, dtype=np.float32).copy()
    entropy_history = []

    # Compute adaptive threshold based on this signal's entropy distribution
    init_map, _ = compute_entropy_map(current_signal)
    adaptive_threshold = float(np.percentile(init_map, 30))
    print(f"  Adaptive threshold set to: {adaptive_threshold:.4f}")

    for evolution in range(num_evolutions):

        # Measure entropy across the current signal
        entropy_map, positions = compute_entropy_map(current_signal, window_size, hop_size)
        avg_entropy = np.mean(entropy_map)
        entropy_history.append(avg_entropy)
        print(f"  Evolution {evolution + 1}/{num_evolutions} — Avg Entropy: {avg_entropy:.4f}")

        # Convert entropy to gain and apply it
        gain = entropy_to_gain(entropy_map, adaptive_threshold, attenuation)
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
import time

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