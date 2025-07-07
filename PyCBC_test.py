import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, spectrogram
import time

# PyCBC imports for realistic templates
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries

# ============================
# LOAD REAL STRAIN DATA
# ============================
input_path = r"C:\Users\kasim\Downloads\H-H1_GWOSC_16KHZ_R1-1268903496-32.txt"
strain = np.loadtxt(input_path)

fs = 16384  # Hz
duration = 32  # seconds
t = np.linspace(0, duration, len(strain), endpoint=False)
print(f"Loaded strain data: {len(strain)} samples at {fs} Hz over {duration} s")

# ============================
# GRAVITATIONAL WAVE PROCESSING
# ============================

def preprocess_strain(strain, fs, low_freq=35, high_freq=500):
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(6, [low, high], btype='band')
    filtered_strain = filtfilt(b, a, strain)
    return filtered_strain

def whiten_strain(strain, fs, seglen=8):
    f, psd = signal.welch(strain, fs, nperseg=seglen*fs, noverlap=seglen*fs//2)
    freqs = np.fft.rfftfreq(len(strain), 1/fs)
    psd_interp = np.interp(freqs, f, psd)
    strain_fft = np.fft.rfft(strain)
    whitened_fft = strain_fft / (np.sqrt(psd_interp) + 1e-12)
    whitened_strain = np.fft.irfft(whitened_fft, n=len(strain))
    return whitened_strain

def detect_chirp_candidates(strain, fs, window_duration=4.0, overlap=0.5, snr_threshold=5.0):
    window_samples = int(window_duration * fs)
    step_samples = int(window_samples * (1 - overlap))
    candidates = []

    for i in range(0, len(strain) - window_samples, step_samples):
        window_data = strain[i:i + window_samples]
        window_time = t[i:i + window_samples]
        first_half = window_data[:len(window_data)//2]
        second_half = window_data[len(window_data)//2:]
        rms_first = np.sqrt(np.mean(first_half**2))
        rms_second = np.sqrt(np.mean(second_half**2))
        energy_ratio = rms_second / (rms_first + 1e-20)
        max_amp = np.max(np.abs(window_data))
        chirp_metric = energy_ratio * max_amp * 1e21
        if chirp_metric > snr_threshold:
            candidates.append({
                'start_time': window_time[0],
                'end_time': window_time[-1],
                'start_idx': i,
                'end_idx': i + window_samples,
                'metric': chirp_metric,
                'data': window_data
            })
    return candidates

def extract_chirp_with_matched_filter(strain, fs, template_duration=0.5):
    template_samples = int(template_duration * fs)
    template_t = np.linspace(0, template_duration, template_samples)
    f_start = 50  # Hz
    f_end = 250   # Hz
    freq_sweep = f_start + (f_end - f_start) * (template_t / template_duration)**2
    template = np.sin(2 * np.pi * np.cumsum(freq_sweep) * template_duration / template_samples)
    window = signal.windows.tukey(len(template), alpha=0.1)
    template = template * window
    correlation = signal.correlate(strain, template, mode='same')
    return correlation, template

# ============================
# NEW: Generate realistic CBC template using PyCBC
# ============================
def generate_realistic_template(fs, duration=0.5):
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=35, mass2=30,
                             delta_t=1/fs,
                             f_lower=20)
    template = hp.numpy()
    desired_samples = int(duration * fs)
    if len(template) > desired_samples:
        template = template[:desired_samples]
    else:
        template = np.pad(template, (0, desired_samples - len(template)))
    window = signal.windows.tukey(len(template), alpha=0.1)
    template *= window
    return template

def matched_filter_frequency_domain(whitened_strain, template, fs):
    if len(template) < len(whitened_strain):
        template = np.pad(template, (0, len(whitened_strain) - len(template)))
    else:
        template = template[:len(whitened_strain)]
    strain_fft = np.fft.fft(whitened_strain)
    template_fft = np.conj(np.fft.fft(template))
    product = strain_fft * template_fft
    corr = np.fft.ifft(product)
    return np.abs(corr)

# ============================
# PROCESS THE DATA
# ============================
start_time = time.time()

print("Step 1: Preprocessing strain data...")
filtered_strain = preprocess_strain(strain, fs)

print("Step 2: Whitening strain data...")
whitened_strain = whiten_strain(filtered_strain, fs)

print("Step 3: Detecting chirp candidates...")
candidates = detect_chirp_candidates(whitened_strain, fs)

print("Step 4: Simple synthetic matched filter...")
correlation, template = extract_chirp_with_matched_filter(whitened_strain, fs)

print("Step 5: Realistic CBC matched filter (PyCBC)...")
realistic_template = generate_realistic_template(fs, duration=0.5)
realistic_corr = matched_filter_frequency_domain(whitened_strain, realistic_template, fs)

end_time = time.time()

# ============================
# RESULTS
# ============================
print(f"\nFound {len(candidates)} potential chirp candidates")
print(f"Processing time: {end_time - start_time:.2f} seconds")
if candidates:
    candidates.sort(key=lambda x: x['metric'], reverse=True)
    best = candidates[0]
    print(f"Top candidate: {best['start_time']:.2f} - {best['end_time']:.2f} s, Metric: {best['metric']:.2f}")

# ============================
# VISUALIZATION
# ============================
fig, axes = plt.subplots(5, 1, figsize=(15, 15))

axes[0].plot(t, strain, color='gray', alpha=0.7)
axes[0].set_title("Original Strain Data")

axes[1].plot(t, filtered_strain, color='blue')
axes[1].set_title("Bandpass Filtered Strain")

axes[2].plot(t, whitened_strain, color='red')
axes[2].set_title("Whitened Strain")
for c in candidates[:3]:
    axes[2].axvspan(c['start_time'], c['end_time'], alpha=0.3, color='yellow')

axes[3].plot(t, np.abs(correlation), color='green')
axes[3].set_title("Synthetic Chirp Matched Filter Output")

axes[4].plot(t, realistic_corr, color='purple')
axes[4].set_title("Realistic CBC Matched Filter Output")

plt.tight_layout()
plt.show()

# ============================
# SPECTROGRAM
# ============================
print("\nGenerating spectrogram...")
f_spec, t_spec, Sxx = spectrogram(whitened_strain, fs, nperseg=1024, noverlap=512)
plt.figure(figsize=(12, 6))
plt.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Whitened Strain Data')
plt.ylim(0, 500)
for c in candidates[:3]:
    plt.axvspan(c['start_time'], c['end_time'], alpha=0.3, color='red')
plt.show()

# ============================
# EXTRACT BEST CANDIDATE
# ============================
if candidates:
    best = candidates[0]
    chirp_data = best['data']
    chirp_time = np.linspace(best['start_time'], best['end_time'], len(chirp_data))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(chirp_time, chirp_data, 'b-', label='Candidate')
    plt.title("Best Chirp Candidate")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    template_time = np.linspace(0, 0.5, len(realistic_template))
    plt.plot(template_time, realistic_template, 'r-', label='Realistic CBC Template')
    plt.title("CBC Template (PyCBC)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nNote: The realistic CBC matched filter uses an actual BBH waveform approximant from PyCBC.")
print("For higher confidence, build a template bank with multiple masses/spins.")

