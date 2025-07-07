from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from scipy.signal import spectrogram, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# Load GPS time
gps = event_gps("GW150914")
start = int(gps) - 4
end   = int(gps) + 4

# 16 kHz strain
strain = TimeSeries.fetch_open_data('H1', start, end, sample_rate=16384)

# ✅ Bandpass 30–350 Hz
Fs = strain.sample_rate.value

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(30, 350, Fs, order=4)
strain_bp = filtfilt(b, a, strain.value)

# ✅ Also whiten
strain_whiten = strain.bandpass(30, 350).whiten(fftlength=4).value

# Choose which to plot: whitened or just bandpassed
signal = strain_whiten

# STFT
nperseg = 4096
noverlap = int(nperseg * 0.75)

f, t, Sxx = spectrogram(signal, fs=Fs, window='blackman',
                        nperseg=nperseg, noverlap=noverlap)

Sxx_dB = 10 * np.log10(Sxx + 1e-20)

plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
plt.ylim(20, 500)
plt.xlabel("Time [s] since chunk start")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram of GW150914 -Hanford")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
plt.show()
