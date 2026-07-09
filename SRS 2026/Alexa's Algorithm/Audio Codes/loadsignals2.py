import os
import librosa
import numpy as np

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
NOISY_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"
DENOISED_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\denoised_output"
CACHE_PATH    = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\cache"
SAMPLE_RATE   = 22050

os.makedirs(CACHE_PATH, exist_ok=True)

# -------------------------------------------------------
# LOAD AND SAVE FUNCTION
# -------------------------------------------------------
def load_and_cache(folder_path, cache_name):
    signals = []
    names = []

    print(f"Loading from {folder_path}...")
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)
            signal, sr = librosa.load(path, sr=SAMPLE_RATE)
            signals.append(signal)
            names.append(file)

    # Save signals as a numpy cache file
    cache_file = os.path.join(CACHE_PATH, cache_name)
    np.savez(cache_file,
             signals=np.array(signals, dtype=object),
             names=np.array(names))

    print(f"  Cached {len(signals)} signals → {cache_name}.npz")
    return signals, names

# -------------------------------------------------------
# RUN ALL THREE
# -------------------------------------------------------
clean_data,    clean_names    = load_and_cache(CLEAN_PATH,    "clean_cache")
noisy_data,    noisy_names    = load_and_cache(NOISY_PATH,    "noisy_cache")
denoised_data, denoised_names = load_and_cache(DENOISED_PATH, "denoised_cache")

print("\nAll signals cached and ready!")
print(f"Cache saved to: {CACHE_PATH}")