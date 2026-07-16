"""
Using PyCBC's own FFT to make:
  (1) amplitude vs time      -> just plot the TimeSeries
  (2) amplitude vs frequency -> TimeSeries.to_frequencyseries(), then abs()

TimeSeries.to_frequencyseries() calls pycbc.fft under the hood and returns a
FrequencySeries that already carries its frequency axis (.sample_frequencies)
and resolution (.delta_f). The low-level equivalent (pycbc.fft.fft) is shown
at the bottom for reference.
"""
# FFT rewrites the strain as a sum of complex sinusoids: each frequency bin holds an amplitude and a phase.
# The inverse FFT is just a reconstruction.
# The round trip check is meaningful sanity check because it proves that the transform is consistent.

from pycbc.waveform import get_td_waveform
import matplotlib
matplotlib.use("Agg")          # headless backend: render to file, no GUI window needed
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")   # silence PyCBC/matplotlib deprecation chatter

# ------------------------------------------------------------------
# A signal to analyse (a TimeSeries)
# ------------------------------------------------------------------
# Generate a time-domain gravitational wave signal from an inspiral merger ringdown waveform model.
ts, _ = get_td_waveform(
    approximant="SEOBNRv4",   # effective-one-body waveform model (aligned-spin)
    mass1=85, mass2=65,       # component masses in solar masses
    distance=400,             # luminosity distance in Mpc (sets overall amplitude)
    delta_t=1.0 / 4096,       # time step -> 4096 Hz sample rate
    f_lower=20.0,             # start the waveform at 20 Hz (LIGO sensitivity floor)
)

# ------------------------------------------------------------------
# (2) Frequency domain via PyCBC's FFT
# ------------------------------------------------------------------
fseries = ts.to_frequencyseries()          # PYCBC FFT
freqs   = fseries.sample_frequencies        # frequency axis (Hz)
fft_amp = abs(fseries)                      # magnitude of each complex bin = spectral amplitude

# len(fseries) = number of positive-frequency bins (real-input rFFT layout).
# delta_f = 1 / signal_duration: longer signals give finer frequency resolution.
print(f"FFT length      : {len(fseries)} bins")
print(f"Freq resolution : {fseries.delta_f:.3f} Hz")
# Index of the loudest bin -> its frequency. For a chirp the FFT amplitude peaks
# low (~20 Hz here) because the inspiral spends the most time at low frequency
# (energy density scales like f^-7/6), not at the merger frequency.
print(f"Peak frequency  : {freqs[fft_amp.numpy().argmax()]:.1f} Hz")

# ------------------------------------------------------------------
# (3) Inverse FFT via PyCBC — back to time domain
# ------------------------------------------------------------------
ts_ifft = fseries.to_timeseries()    # PYCBC inverse FFT: FrequencySeries -> TimeSeries

# Sanity check: an FFT followed by its inverse should return the original signal.
# Subtract sample-by-sample (trim to original length in case IFFT padded/rounded)
# and report the largest deviation. Expect ~1e-13 — i.e. floating-point round-off,
# confirming the transform pair is numerically consistent.
residual = ts.numpy() - ts_ifft.numpy()[:len(ts)]
print(f"Max reconstruction error: {abs(residual).max():.3e}")

# ------------------------------------------------------------------
# Plot: amplitude vs time (left)  and  amplitude vs frequency (right)
# ------------------------------------------------------------------
# Three side-by-side panels: original time series, its spectrum, and the
# IFFT-reconstructed time series (should look identical to the first panel).
fig, (ax_t, ax_f, ax_i) = plt.subplots(1, 3, figsize=(18, 5))

# Left: the raw strain waveform. sample_times supplies the time axis in seconds.
ax_t.plot(ts.sample_times, ts, color="steelblue", lw=0.9)
ax_t.set(title="Amplitude vs Time (original)", xlabel="Time (s)", ylabel="Strain")
ax_t.grid(alpha=0.3)

# Middle: spectral amplitude on a log-y scale (semilogy) so the dynamic range
# across frequencies is visible. Zoomed to the 10–150 Hz band of interest.
ax_f.semilogy(freqs, fft_amp, color="darkorange", lw=0.9)
ax_f.set(title="Amplitude vs Frequency (PyCBC FFT)",
         xlabel="Frequency (Hz)", ylabel="|FFT| amplitude", xlim=(10, 150))
ax_f.grid(alpha=0.3, which="both")

# Right: the reconstructed waveform from the inverse FFT (visual confirmation
# of the round-trip check printed above).
ax_i.plot(ts_ifft.sample_times, ts_ifft, color="seagreen", lw=0.9)
ax_i.set(title="Amplitude vs Time (PyCBC IFFT)", xlabel="Time (s)", ylabel="Strain")
ax_i.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pycbc_fft_ifft_plots.png", dpi=130, bbox_inches="tight")
print("Figure saved.")
