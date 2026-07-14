"""
fourier_pipeline.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma
from pycbc.filter import highpass, lowpass

# ---- Random number generator ----
SEED = 4                      # set to an int for reproducible runs
rng = np.random.default_rng(SEED)

# ---- Source parameters ----
APPROXIMANT = "NRSur7dq4"
SPIN1Z, SPIN2Z = 0.2, -0.1
DISTANCE = 400.0        # Mpc
DELTA_T = 1.0 / 4096.0  # s
F_LOWER = 20.0          # Hz
TARGET_SNR = 20.0       # tunable network matched-filter SNR
PSD_LOW_FREQ = 20.0     # Hz; lower edge for SNR/noise, matches F_LOWER
BANDPASS_LOW = 30.0     # Hz; low edge of the passband
BANDPASS_HIGH = 400.0   # Hz; high edge of the passband
WHITEN_LOW_FREQ = BANDPASS_LOW   # only whiten within the passband
# --- masses: draw total mass and mass ratio inside NRSur7dq4's valid region ---
total_mass = rng.uniform(60.0, 120.0)   # Msun; >=60 keeps f_lower=20 Hz valid
mass_ratio = rng.uniform(1.0, 3.0)      # q = m1/m2, NRSur7dq4 trained up to ~4
MASS1 = total_mass * mass_ratio / (1.0 + mass_ratio)   # heavier component
MASS2 = total_mass / (1.0 + mass_ratio)                # lighter component

INCLINATION = np.arccos(rng.uniform(-1.0, 1.0))  # isotropic orientation, 0..pi
COA_PHASE = rng.uniform(0.0, 2.0 * np.pi)        # coalescence phase

# ---- Sky location (fixed) ----
DETECTOR = "H1"
GPS_TIME = 1126259462.0

# ---- detectors used only to test that a sky angle gives good response in BOTH ----
det_h = Detector("H1")
det_l = Detector("L1")
MIN_RESPONSE = 0.55   # require |F| = sqrt(F+^2 + Fx^2) >= this in each detector

while True:
    RIGHT_ASCENSION = rng.uniform(0.0, 2.0 * np.pi)      # RA, uniform on the circle
    DECLINATION = np.arcsin(rng.uniform(-1.0, 1.0))      # Dec, isotropic on the sphere
    POLARIZATION = rng.uniform(0.0, np.pi)               # polarization angle
    resp_h = np.hypot(*det_h.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME))
    resp_l = np.hypot(*det_l.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME))
    if resp_h >= MIN_RESPONSE and resp_l >= MIN_RESPONSE:
        break
print(f"M1={MASS1:.1f}  M2={MASS2:.1f}  q={mass_ratio:.2f}  incl={INCLINATION:.2f}")
print(f"RA={RIGHT_ASCENSION:.2f}  Dec={DECLINATION:.2f}  psi={POLARIZATION:.2f}  "
      f"|F_H|={resp_h:.2f}  |F_L|={resp_l:.2f}")

# ---- Generate the two polarizations ----
hp, hc = get_td_waveform(
    approximant=APPROXIMANT,
    mass1=MASS1, mass2=MASS2,
    spin1z=SPIN1Z, spin2z=SPIN2Z,
    distance=DISTANCE, inclination=INCLINATION, coa_phase=COA_PHASE,
    delta_t=DELTA_T, f_lower=F_LOWER, f_ref=F_LOWER,
)

time = hp.sample_times.numpy()
h_plus = hp.numpy()
h_cross = hc.numpy()

# ---- Project onto Hanford: h(t) = F+ h+ + Fx hx ----
f_plus, f_cross = Detector(DETECTOR).antenna_pattern(
    RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)
detector_strain = f_plus * h_plus + f_cross * h_cross

# ---- Project onto Livingston: h(t) = F+ h+ + Fx hx ----
detector_l = Detector("L1")
f_plus_l, f_cross_l = detector_l.antenna_pattern(
    RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)
detector_strain_l = f_plus_l * h_plus + f_cross_l * h_cross

# ---- Compute time delay between H1 and L1 for this sky location ----
dt_HL = detector_l.time_delay_from_detector(
    Detector("H1"), RIGHT_ASCENSION, DECLINATION, GPS_TIME)
print(f"H->L time delay: {dt_HL*1e3:+.3f} ms")

print(f"{DETECTOR}: F+ = {f_plus:+.4f}, Fx = {f_cross:+.4f}")
print(f"L1: F+ = {f_plus_l:+.4f}, Fx = {f_cross_l:+.4f}")
# ---- Shift time so the merger (peak amplitude) sits at t = 0 ----
peak_index = np.argmax(np.sqrt(h_plus**2 + h_cross**2))
t = time - time[peak_index]
t_l = t + dt_HL

# ---- build detector strains as TimeSeries (hp, hc are TimeSeries) so the frequency-domain tools below work directly ----
strain_h = f_plus   * hp + f_cross   * hc
strain_l = f_plus_l * hp + f_cross_l * hc

# aLIGO design noise curve, sampled to match this data's frequency grid
n = len(strain_h)
delta_f = 1.0 / (n * DELTA_T)
psd = aLIGOZeroDetHighPower(n // 2 + 1, delta_f, PSD_LOW_FREQ)

# optimal matched-filter SNR each detector would have for this signal in this noise
sigma_h = sigma(strain_h, psd=psd, low_frequency_cutoff=PSD_LOW_FREQ)
sigma_l = sigma(strain_l, psd=psd, low_frequency_cutoff=PSD_LOW_FREQ)
network_sigma = np.sqrt(sigma_h**2 + sigma_l**2)

# rescale BOTH strains by the SAME factor so the network SNR equals TARGET_SNR
scale = TARGET_SNR / network_sigma
strain_h *= scale
strain_l *= scale

# draw independent realistic noise for each detector and add it
noise_h = noise_from_psd(n, DELTA_T, psd, seed=int(rng.integers(2**31 - 1)))
noise_l = noise_from_psd(n, DELTA_T, psd, seed=int(rng.integers(2**31 - 1)))
noise_h.start_time = strain_h.start_time
noise_l.start_time = strain_l.start_time
data_h = strain_h + noise_h
data_l = strain_l + noise_l

# ---- Band-pass filter the data to remove low-frequency noise and high-frequency noise ----
def bandpass(series):
    """LIGO-style band-pass: high-pass then low-pass, keeping BANDPASS_LOW..HIGH."""
    hp_filtered = highpass(series, BANDPASS_LOW)              # remove f < 30 Hz
    bp_filtered = lowpass(hp_filtered, BANDPASS_HIGH)         # remove f > 400 Hz
    return bp_filtered
bp_h = bandpass(data_h)
bp_l = bandpass(data_l)
# same band-pass applied to the noise-free signal, for comparison
clean_bp_h = bandpass(strain_h)
clean_bp_l = bandpass(strain_l)

# ---- Whiten the data to flatten the noise spectrum, for easier visual comparison of signal vs noise ----
def whiten(series):
    """Divide by the known noise amplitude (sqrt PSD) inside the band only.
    Operates on numpy arrays to avoid boolean-index and divide-by-zero issues."""
    freq = series.to_frequencyseries()
    asd = np.asarray(psd) ** 0.5              # amplitude spectral density, as numpy
    data = np.asarray(freq)                   # the complex spectrum, as numpy

    # in-band mask: only whiten where the PSD is valid AND above the low cutoff
    f = np.asarray(freq.sample_frequencies)
    in_band = (f >= WHITEN_LOW_FREQ) & (asd > 0) & np.isfinite(asd)

    white = np.zeros_like(data)               # everything out of band stays zero
    white[in_band] = data[in_band] / asd[in_band]   # safe division, no zeros hit

    from pycbc.types import FrequencySeries
    return FrequencySeries(white, delta_f=freq.delta_f,
                           epoch=freq.epoch).to_timeseries(delta_t=series.delta_t)
white_h = whiten(bp_h)
white_l = whiten(bp_l)

print(f"per-detector SNR: H={sigma_h*scale:.1f}, L={sigma_l*scale:.1f}  "
      f"(network = {TARGET_SNR:.1f})")


def qp_wavelet_fd(freqs, tau, nu, Q, p):
    """Frequency-domain Qp-wavelet (conjugate psi-tilde*), Virtuoso & Milotti Eq. 19d.
    freqs: array of frequencies (Hz); tau: center time (s); nu: center freq (Hz);
    Q: quality factor; p: chirp parameter (0 recovers the plain Q-wavelet)."""
    z = 1.0 + 2.0j * Q * p                                  # complex chirp factor (1+2iQp)
    norm = (1.0 / (2.0 * np.pi * nu**2 * Q**2)) ** 0.25      # amplitude normalization
    prefactor = norm * (Q / np.sqrt(z))
    gaussian = np.exp(-((Q /(2.0 * np.sqrt(z))) * (freqs - nu) / nu) ** 2)  # tilted Gaussian in f
    phase = np.exp(2.0j * np.pi * freqs * tau)              # time-localization phase
    return prefactor * gaussian * phase

def qp_frequency_grid(f_min, f_max, Q, p, alpha=0.5):
    """Log-adaptive frequency rows, Virtuoso & Milotti Eq. 24:
    nu_{j+1} = nu_j * (1 + (alpha/Q) * sqrt(1 + (2pQ)^2))."""
    freqs = [f_min]
    step_factor = 1.0 + (alpha / Q) * np.sqrt(1.0 + (2.0 * p * Q) ** 2)
    while freqs[-1] * step_factor < f_max:
        freqs.append(freqs[-1] * step_factor)
    return np.array(freqs)

def qp_transform(strain_td, fs, freqs, Q, p):
    """Qp-transform energy map |T(tau,nu)|^2 via one inverse FFT per frequency row.
    Returns (times, freqs, tf) with tf shaped (n_freq, n_time)."""
    s = np.asarray(strain_td, dtype=float)
    N = s.size
    dt = 1.0 / fs
    s_tilde = np.fft.fft(s)                      # data in the frequency domain
    fft_freqs = np.fft.fftfreq(N, dt)            # its matching frequency axis (Hz)
    times = np.arange(N) * dt                    # uniform full-resolution time grid
    tf = np.empty((freqs.size, N))
    for j, nu in enumerate(freqs):
        W = qp_wavelet_fd(fft_freqs, tau=0.0, nu=nu, Q=Q, p=p)  # wavelet, no time-shift
        T = np.fft.ifft(s_tilde * W)             # inverse FFT gives T(tau,nu) for all tau
        tf[j] = np.abs(T) ** 2                    # energy in this frequency row
    return times, freqs, tf

THRESHOLD = 7.0   # |T|^2 threshold; for chi^2(2,mean 2) noise, 7 -> ~0.1% false alarm
def energy_density(tf, freqs, thr=THRESHOLD, noise_cols=200):
    # estimate noise from the first `noise_cols` time columns (pre-signal),
    # so the normalization doesn't depend on how many tiles the map has
    noise_level = np.median(tf[:, :noise_cols])
    tf_norm = tf * (2.0 / noise_level)
    dnu = np.gradient(freqs)
    weight = np.broadcast_to(dnu[:, None], tf_norm.shape)
    mask = tf_norm > thr
    if not mask.any():
        return 0.0, float(tf_norm.max()), 0
    density = (tf_norm * weight)[mask].sum() / weight[mask].sum()
    return float(density), float(tf_norm.max()), int(mask.sum())

# ---- Find the best Qp-transform parameters for this whitened data ----
fs_val = 1.0 / DELTA_T
Q_FIXED, P_BEST = 6, 0.08
fg = qp_frequency_grid(BANDPASS_LOW, BANDPASS_HIGH, Q_FIXED, P_BEST)
qp_t, qp_f, qp_tf = qp_transform(white_h.numpy(), fs_val, fg, Q=Q_FIXED, p=P_BEST)

# Q-transform: time-frequency power map of each whitened detector strain
qt_low, qt_high = BANDPASS_LOW, BANDPASS_HIGH
t_qh, f_qh, power_h = white_h.qtransform(delta_t=0.002, logfsteps=300,
                                         qrange=(4, 8), frange=(qt_low, qt_high))
t_ql, f_ql, power_l = white_l.qtransform(delta_t=0.002, logfsteps=300,
                                         qrange=(4, 8), frange=(qt_low, qt_high))
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

def center_on_peak(times, tf):
    return times - times[np.argmax(tf.sum(axis=0))]   # put loudest column at t=0

# ---- Plot our Qp-transform (p=0) ----
from matplotlib.colors import LogNorm
figb, axb = plt.subplots(figsize=(10, 5))
tfb = best["tf"] * (2.0 / np.median(best["tf"]))
axb.pcolormesh(best["t"] - best["t"][np.argmax(best["tf"].sum(axis=0))], best["f"], tfb,
               shading="auto", cmap="viridis", norm=LogNorm(vmin=2, vmax=tfb.max()))
axb.set_yscale("log"); axb.set_ylabel("freq (Hz)"); axb.set_xlabel("time (s)")
axb.set_title(f"Best Qp-transform: Q={best['Q']}, p={best['p']}, density={best['density']:.2f}")
figb.tight_layout()

# ---- Plot Q-transform of whitened data for each detector ----
m0 = ax2[0].pcolormesh(t_qh - time[peak_index], f_qh, power_h,
                       shading="auto", cmap="viridis")
ax2[0].legend(loc="upper left")
ax2[0].set_yscale("log")
ax2[0].set_ylabel("frequency (Hz)")
ax2[0].set_title("Hanford (H1) Q-transform")
fig2.colorbar(m0, ax=ax2[0], label="normalized power")

m1 = ax2[1].pcolormesh(t_ql - time[peak_index] + dt_HL, f_ql, power_l,
                       shading="auto", cmap="viridis")
ax2[1].legend(loc="upper left")
ax2[1].set_yscale("log")
ax2[1].set_ylabel("frequency (Hz)")
ax2[1].set_xlabel("time relative to Hanford peak (s)")
ax2[1].set_title("Livingston (L1) Q-transform")
fig2.colorbar(m1, ax=ax2[1], label="normalized power")

fig2.tight_layout()

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

wsh = whiten(strain_h).numpy()
m = min(len(t), len(wsh))
axes[0].plot(t[:m], wsh[:m], color="black", lw=1.5, label="clean")
wh = white_h.numpy()
mh = min(len(t), len(wh))
axes[0].plot(t[:mh], wh[:mh], color="tab:purple", alpha=0.6, label="whitened data")
axes[0].set_ylabel("strain")
axes[0].set_title(f"Hanford (H1)  ($F_+$={f_plus:+.3f}, $F_\\times$={f_cross:+.3f})")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

wsl = whiten(strain_l).numpy()
ml = min(len(t_l), len(wsl))
axes[1].plot(t_l[:ml], wsl[:ml], color="black", lw=1.5, label="clean")
wl = white_l.numpy()
ml = min(len(t_l), len(wl))
axes[1].plot(t_l[:ml], wl[:ml], color="tab:green", alpha=0.6, label="whitened data")
axes[1].set_ylabel("strain")
axes[1].set_xlabel("time relative to Hanford peak (s)")
axes[1].set_title(f"Livingston (L1)  ($F_+$={f_plus_l:+.3f}, $F_\\times$={f_cross_l:+.3f})")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
out = Path("figures") / f"{DETECTOR}_strain.png"
fig.savefig(out, dpi=200)
print(f"Saved {out}")
plt.show()