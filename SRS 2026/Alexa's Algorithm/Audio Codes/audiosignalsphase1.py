import os
import librosa 
import numpy as np

#---SETTINGS---
DATASET_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
SAMPLE_RATE = 22050 # Standard sample rate in Hz

#---LOAD ALL AUDIO FILES---
audio_data = [] # Will store the audio waveforms
file_names = [] # Will store the file names

for file in os.listdir(DATASET_PATH):
	if file.endswith(".wav"):
		file_path = os.path.join(DATASET_PATH, file)
		signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
		audio_data.append(signal)
		file_names.append(file)
		print (f"Loaded: {file} | Length: {len(signal)} samples")

print(f"\nDone! Total files loaded: {len(audio_data)}")

# --- NOISE SETTINGS ---
NOISE_SCALE = 0.05       # Base noise intensity (increase for more noise)
VARIATION_SPEED = 2.0    # How fast the noise intensity changes over time

# --- FUNCTION TO ADD NON-GAUSSIAN NON-STATIC NOISE ---
def add_noise(signal, sample_rate, scale=NOISE_SCALE, speed=VARIATION_SPEED):
    
    # Step 1: Generate Laplacian noise (non-Gaussian)
    laplacian_noise = np.random.laplace(loc=0.0, scale=scale, size=len(signal))
    
    # Step 2: Create a time-varying envelope (non-static)
    time_axis = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * speed * time_axis)  # Oscillates between 0 and 1
    
    # Step 3: Multiply noise by envelope so intensity changes over time
    dynamic_noise = laplacian_noise * envelope
    
    # Step 4: Add the noise to the original signal
    noisy_signal = signal + dynamic_noise
    
    return noisy_signal, dynamic_noise

# --- APPLY NOISE TO ALL LOADED SIGNALS ---
noisy_data = []

for i, signal in enumerate(audio_data):
    noisy_signal, noise = add_noise(signal, SAMPLE_RATE)
    noisy_data.append(noisy_signal)
    print(f"Noise added to: {file_names[i]}")

print(f"\nDone! {len(noisy_data)} noisy signals created.")


import soundfile as sf
import os

# --- SAVE NOISY SIGNALS TO DISK ---
OUTPUT_PATH = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\noisy_output"

try:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Folder confirmed at: {OUTPUT_PATH}")

    for i, noisy_signal in enumerate(noisy_data):
        output_file = os.path.join(OUTPUT_PATH, f"noisy_{file_names[i]}")
        sf.write(output_file, noisy_signal, SAMPLE_RATE)
        print(f"Saved: {output_file}")

    print(f"\nDone! {len(noisy_data)} noisy files saved.")

except Exception as e:
    print(f"ERROR: Something went wrong — {e}")