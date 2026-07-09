import numpy as np
import os

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH    = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/clean"
NOISY_PATH    = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/noisy"
DENOISED_PATH = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/denoised"
CACHE_PATH    = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/cache"

os.makedirs(CACHE_PATH, exist_ok=True)

# -------------------------------------------------------
# LOAD AND CACHE FUNCTION
# -------------------------------------------------------
def load_and_cache(folder_path, cache_name):
    signals = []
    names   = []

    files = sorted([f for f in os.listdir(folder_path)
                    if f.endswith(".npy") and f != "metadata.npy"])

    print(f"Loading {len(files)} signals from {folder_path}...")

    for filename in files:
        path   = os.path.join(folder_path, filename)
        signal = np.load(path).astype(np.float32)
        signals.append(signal)
        names.append(filename.replace(".npy", ""))

    cache_file = os.path.join(CACHE_PATH, cache_name)
    np.savez(cache_file,
             signals = np.array(signals, dtype=object),
             names   = np.array(names))

    print(f"  Cached {len(signals)} signals → {cache_name}.npz")
    return signals, names


# -------------------------------------------------------
# CACHE METADATA
# -------------------------------------------------------
def cache_metadata(source_path, cache_name):
    meta_path = os.path.join(source_path, "metadata.npy")
    if os.path.exists(meta_path):
        metadata = np.load(meta_path, allow_pickle=True).item()
        out_path = os.path.join(CACHE_PATH, cache_name)
        np.save(out_path, metadata)
        print(f"  Metadata cached → {cache_name}.npy")
    else:
        print(f"  No metadata found in {source_path} — skipping.")


# -------------------------------------------------------
# RUN ALL THREE
# -------------------------------------------------------
clean_data,    clean_names    = load_and_cache(CLEAN_PATH,    "clean_cache")
cache_metadata(CLEAN_PATH, "clean_metadata")

noisy_data,    noisy_names    = load_and_cache(NOISY_PATH,    "noisy_cache")
cache_metadata(NOISY_PATH, "noisy_metadata")

# Denoised folder may not exist yet on first run — skip gracefully
if os.path.exists(DENOISED_PATH) and any(
        f.endswith(".npy") for f in os.listdir(DENOISED_PATH)):
    denoised_data, denoised_names = load_and_cache(DENOISED_PATH, "denoised_cache")
    cache_metadata(DENOISED_PATH, "denoised_metadata")
else:
    print(f"\nDenoised folder empty or not found — skipping denoised cache.")
    print(f"Run gwdenoise.py first, then re-run this script.")

print(f"\nAll available signals cached.")
print(f"Cache saved to: {CACHE_PATH}")
