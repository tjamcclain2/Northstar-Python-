import numpy as np
import os
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.filter import highpass

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH  = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/clean"
NOISY_PATH  = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/noisy"
SAMPLE_RATE = 4096
F_LOWER     = 20.0       # Hz — highpass cutoff, matches chirp generation
NOISE_SCALE = 1.0        # Multiply noise amplitude (1.0 = realistic LIGO noise level)
                         # Increase to make denoising harder, decrease to make it easier

# -------------------------------------------------------
# NOISE SOURCE
# -------------------------------------------------------
# This function is the swappable noise source.
# Currently: generates colored Gaussian noise using the aLIGO design PSD.
# Future swap: replace with a function that loads a real LIGO noise segment
#              from a GWF frame file using pycbc.frame.read_frame().
# The interface stays the same — it returns a numpy array of noise samples
# the same length as the clean signal.

def generate_noise(n_samples, sample_rate, scale=NOISE_SCALE):

    delta_f   = 1.0 / (n_samples / sample_rate)
    flen      = n_samples // 2 + 1

    # Build the aLIGO design sensitivity PSD
    psd = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    # Generate colored Gaussian noise from the PSD
    noise_ts = noise_from_psd(n_samples, 1.0 / sample_rate, psd)

    noise = np.array(noise_ts.data, dtype=np.float64)

    # Normalize noise to have unit variance, then apply scale
    std = np.std(noise)
    if std > 0:
        noise = (noise / std) * scale

    return noise


# -------------------------------------------------------
# HIGHPASS FILTER
# -------------------------------------------------------
# Removes sub-20Hz content where LIGO has no sensitivity
# Applied to both signal and noise before injection
def apply_highpass(signal, sample_rate, f_lower=F_LOWER):
    ts = TimeSeries(signal, delta_t=1.0/sample_rate)
    ts = highpass(ts, frequency=f_lower)
    return np.array(ts.data, dtype=np.float64)


# -------------------------------------------------------
# INJECT NOISE INTO CLEAN SIGNALS
# -------------------------------------------------------
os.makedirs(NOISY_PATH, exist_ok=True)

metadata   = np.load(os.path.join(CLEAN_PATH, "metadata.npy"), allow_pickle=True).item()
signal_files = [f for f in sorted(os.listdir(CLEAN_PATH)) if f.endswith(".npy") and f != "metadata.npy"]

print(f"Found {len(signal_files)} clean signals.")
print(f"Saving noisy signals to: {NOISY_PATH}\n")

noisy_metadata = {}

for filename in signal_files:
    signal_name = filename.replace(".npy", "")
    clean_path  = os.path.join(CLEAN_PATH, filename)
    noisy_path  = os.path.join(NOISY_PATH, filename)

    clean  = np.load(clean_path)
    n      = len(clean)

    print(f"Processing: {signal_name}  ({n} samples)")

    # Apply highpass to clean signal
    clean_filtered = apply_highpass(clean, SAMPLE_RATE)

    # Generate colored noise matching the signal length
    noise = generate_noise(n, SAMPLE_RATE)

    # Normalize noise relative to signal peak so injection SNR is consistent
    signal_peak = np.max(np.abs(clean_filtered))
    noise_peak  = np.max(np.abs(noise))
    if noise_peak > 0:
        noise = noise * (signal_peak / noise_peak) * NOISE_SCALE

    # Inject: noisy = clean + noise
    noisy = clean_filtered + noise

    np.save(noisy_path, noisy.astype(np.float64))

    # Carry over metadata and add noise info
    if signal_name in metadata:
        noisy_metadata[signal_name] = dict(metadata[signal_name])
        noisy_metadata[signal_name]["noise_scale"] = NOISE_SCALE
        noisy_metadata[signal_name]["noise_type"]  = "aLIGOZeroDetHighPower_colored_gaussian"

    print(f"  Saved: {filename}")

np.save(os.path.join(NOISY_PATH, "metadata.npy"), noisy_metadata)
print(f"\nDone! {len(signal_files)} noisy signals saved.")
print(f"Metadata saved to: {NOISY_PATH}/metadata.npy")
