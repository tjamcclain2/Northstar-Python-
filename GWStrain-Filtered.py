# --------------------------------------------------------
# ✅ GWpy + scipy: Real LIGO strain → STFT → filter → ISTFT
# --------------------------------------------------------

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from scipy.signal import stft, istft, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1️⃣ Get GW strain: e.g., GW150914 Hanford
# --------------------------------------------------------

event = "GW150914"
gps = event_gps(event)
print(f"{event} GPS: {gps}")

detector = "H1"
start = int(gps) - 4
end   = int(gps) + 4

Fs = 16384
strain = TimeSeries.fetch_open_data(detector, start, end, sample_rate=Fs)

# --------------------------------------------------------
# 2️⃣ Bandpass with Butterworth (30–350 Hz)
# --------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(30, 350, Fs, order=4)
strain_bp = filtfilt(b, a, strain.value)

# --------------------------------------------------------
# 3️⃣ Whiten (using GWpy)
# --------------------------------------------------------

strain_whitened = strain.bandpass(30, 350).whiten(fftlength=4).value

# For STFT: pick the bandpassed version
signal = strain_whitened

# --------------------------------------------------------
# 4️⃣ Run STFT
# --------------------------------------------------------

nperseg = 4096
noverlap = int(nperseg * 0.75)

f, t, Zxx = stft(signal, fs=Fs, window='blackman',
                 nperseg=nperseg, noverlap=noverlap)

# --------------------------------------------------------
# 5️⃣ Denoise: Example → zero out bins above 400 Hz
# --------------------------------------------------------

Zxx_denoised = Zxx.copy()
Zxx_denoised[f > 400, :] = 0

# --------------------------------------------------------
# 6️⃣ Invert back → ISTFT
# --------------------------------------------------------

_, signal_reconstructed = istft(Zxx_denoised, fs=Fs, window='blackman',
                                nperseg=nperseg, noverlap=noverlap)

# --------------------------------------------------------
# 7️⃣ Plot: Original vs. Reconstructed
# --------------------------------------------------------

t_arr = np.linspace(0, len(signal)/Fs, len(signal))

plt.figure(figsize=(12, 6))
plt.plot(t_arr, signal, label='Original (whitened, BP)')
plt.plot(t_arr[:len(signal_reconstructed)], signal_reconstructed, label='Reconstructed (STFT filtered)', alpha=0.7)
plt.plot(t_arr, strain.value[:len(t_arr)], label="Raw strain")
plt.xlabel("Time [s] since chunk start")
plt.ylabel("Strain [whitened]")
plt.title(f"{detector} {event}: Time Domain After STFT → Filter → ISTFT")
plt.legend()
plt.tight_layout()
plt.show()
