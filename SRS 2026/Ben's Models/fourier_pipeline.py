"""
fourier_pipeline.py

Simulate a two-detector (Hanford + Livingston) gravitational-wave observation and
process it with a wavelet Qp-transform to locate the merger in time-frequency.

The Qp transform used in this pipeline is a chirping-wavelet transform that is more sensitive to the
inspiral phase of a binary merger than the standard Q-transform. It is based on the work of
Virtuoso & Milotti, "Chirplet-based analysis of gravitational-wave signals",
https://arxiv.org/pdf/2404.18781, which I reccomend reading for a deeper understanding of the Qp transform 
and its advantages over the standard Q-transform. I do not completely recreate the Qp transform here,
but I do implement the core equations and use it to extract the merger time from a simulated signal. 

Pipeline stages (in execution order below):
    1. Draw random source parameters (masses, inclination, phase) in NRSur7dq4's range.
    2. Draw a random sky location where BOTH detectors respond well.
    3. Generate the NRSur7dq4 polarizations h+, hx.
    4. Project onto each detector via its antenna pattern; compute the H-L time delay.
    5. Inject the signal into aLIGO-coloured noise at a chosen network SNR.
    6. Band-pass and whiten the data (standard LIGO-style preprocessing).
    7. Compute the Qp-transform (chirping-wavelet TF map) and estimate the merger time.
    8. Plot the Q-transforms and the whitened strains.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma, highpass, lowpass
from pycbc.types import FrequencySeries
from pycbc.types import TimeSeries


# ============================================================
# CONFIGURATION
# ============================================================

# --- Reproducibility ---
SEED = 4                                 # set to an int for a repeatable run
rng = np.random.default_rng(SEED)

# --- Waveform / source ---
APPROXIMANT = "NRSur7dq4"
SPIN1Z, SPIN2Z = 0.2, -0.1               # aligned spins (no in-plane spin for now)
DISTANCE = 400.0                         # Mpc
DELTA_T = 1.0 / 4096.0                   # s (sample spacing)
F_LOWER = 20.0                           # Hz (starting frequency)

# --- Detectors / sky ---
DETECTOR = "H1"                          # reference detector for the plots
GPS_TIME = 1126259462.0                  # event time (sets detector orientation)
MIN_RESPONSE = 0.55                      # require |F| >= this in BOTH detectors

# --- Noise injection ---
TARGET_SNR = 20.0                        # network matched-filter SNR to inject
PSD_LOW_FREQ = 20.0                      # Hz (low edge for SNR/noise; matches F_LOWER)

# --- Preprocessing (band-pass + whiten) ---
BANDPASS_LOW = 30.0                      # Hz (passband low edge)
BANDPASS_HIGH = 400.0                    # Hz (passband high edge)
WHITEN_LOW_FREQ = BANDPASS_LOW           # only whiten inside the passband

# --- Qp-transform ---
FS = 1.0 / DELTA_T                       # Hz (sample rate)
Q_FIXED = 6                              # low Q favours time resolution (good for timing)
P_BEST = 0.08                            # chirp parameter (energy-density optimum for this Q)
THRESHOLD = 7.0                          # |T|^2 threshold; chi^2(2,mean 2) -> ~0.1% false alarm


# ============================================================
# FUNCTIONS
# ============================================================

def bandpass(series):
    """LIGO-style band-pass: zero-phase high-pass then low-pass, keeping
    BANDPASS_LOW..BANDPASS_HIGH. Zero-phase so the merger is not time-shifted."""
    high_passed = highpass(series, BANDPASS_LOW)          # remove f < BANDPASS_LOW
    band_passed = lowpass(high_passed, BANDPASS_HIGH)     # remove f > BANDPASS_HIGH
    return band_passed


def whiten(series):
    """Divide the spectrum by the noise amplitude (sqrt of the module-level PSD),
    inside the passband only, to flatten the noise. Works on numpy arrays to avoid
    boolean-index and divide-by-zero pitfalls with pycbc FrequencySeries."""
    freq = series.to_frequencyseries()
    asd = np.asarray(psd) ** 0.5                          # amplitude spectral density
    spectrum = np.asarray(freq)                           # complex spectrum

    # Whiten only where the PSD is valid and above the low cutoff; zero elsewhere.
    f = np.asarray(freq.sample_frequencies)
    in_band = (f >= WHITEN_LOW_FREQ) & (asd > 0) & np.isfinite(asd)

    white = np.zeros_like(spectrum)
    white[in_band] = spectrum[in_band] / asd[in_band]

    return FrequencySeries(white, delta_f=freq.delta_f,
                           epoch=freq.epoch).to_timeseries(delta_t=series.delta_t)


def qp_wavelet_fd(freqs, tau, nu, Q, p):
    """Frequency-domain Qp-wavelet (conjugate psi-tilde*), Virtuoso & Milotti Eq. 19d.

    freqs : frequency axis (Hz)      nu : wavelet centre frequency (Hz)
    tau   : wavelet centre time (s)  Q  : quality factor
    p     : chirp parameter (p = 0 recovers the plain Q-wavelet)
    """
    z = 1.0 + 2.0j * Q * p                                # complex chirp factor (1 + 2iQp)
    norm = (1.0 / (2.0 * np.pi * nu ** 2 * Q ** 2)) ** 0.25
    prefactor = norm * (Q / np.sqrt(z))
    gaussian = np.exp(-((Q / (2.0 * np.sqrt(z))) * (freqs - nu) / nu) ** 2)
    phase = np.exp(2.0j * np.pi * freqs * tau)            # time-localization phase
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
    """Qp-transform energy map |T(tau, nu)|^2, one inverse FFT per frequency row.

    The wavelet's only tau-dependence is a factor e^{2*pi*i*f*tau}, so for a fixed
    nu the transform over all tau is a single inverse FFT of s_tilde(f) * W(f; nu).
    Returns (times, freqs, tf) with tf shaped (n_freq, n_time).
    """
    s = np.asarray(strain_td, dtype=float)
    N = s.size
    dt = 1.0 / fs

    s_tilde = np.fft.fft(s)                                # data in the frequency domain
    fft_freqs = np.fft.fftfreq(N, dt)                     # matching frequency axis (Hz)
    times = np.arange(N) * dt                             # full-resolution time grid

    tf = np.empty((freqs.size, N))
    for j, nu in enumerate(freqs):
        W = qp_wavelet_fd(fft_freqs, tau=0.0, nu=nu, Q=Q, p=p)
        T = np.fft.ifft(s_tilde * W)                      # T(tau, nu) for all tau
        tf[j] = np.abs(T) ** 2                            # energy in this frequency row
    return times, freqs, tf


def energy_density(tf, freqs, thr=THRESHOLD, noise_cols=200):
    """Paper Eq. 30: area-weighted mean energy of the above-threshold tiles.
    Higher = signal packed into fewer, brighter tiles (a more compact representation).
    Returns (density, peak, n_tiles). Noise level is estimated from the first
    `noise_cols` (pre-signal) time columns so the normalization is grid-independent."""
    noise_level = np.median(tf[:, :noise_cols])
    tf_norm = tf * (2.0 / noise_level)                    # normalize noise floor to mean ~2

    dnu = np.gradient(freqs)                              # frequency width of each row
    weight = np.broadcast_to(dnu[:, None], tf_norm.shape) # tile-area weight (dtau cancels)

    mask = tf_norm > thr
    if not mask.any():
        return 0.0, float(tf_norm.max()), 0
    density = (tf_norm * weight)[mask].sum() / weight[mask].sum()
    return float(density), float(tf_norm.max()), int(mask.sum())


def merger_time_from_energy(times, freqs, tf, thr=THRESHOLD, noise_cols=200,
                            window_s=0.03):
    """Merger time = time of the energy PEAK of the thresholded Qp map. Threshold
    at E_thr (chi^2 noise), then use the energy peak as the signal locator. We take 
    the energy-weighted centroid in a +/- window_s window around that peak, so the
    long inspiral cannot bias the estimate and sub-bin precision is retained.
    Uses energy, so ringdown high-frequency content does not pull
    the estimate late. Returns (t_merger, t_sigma, power_t).
    """
    noise_level = np.median(tf[:, :noise_cols])
    tf_norm = tf * (2.0 / noise_level)                 # chi^2 mean-2 normalization

    tf_thr = np.where(tf_norm > thr, tf_norm, 0.0)     # paper's E_thr thresholding
    power_t = tf_thr.sum(axis=0)                        # surviving energy vs time
    if power_t.sum() == 0:
        raise ValueError("no tiles above threshold; lower THRESHOLD or check the map")

    # locate the energy peak (the paper's signal descriptor), then centroid nearby
    peak_idx = int(np.argmax(power_t))
    dt = times[1] - times[0]
    half = int(round(window_s / dt))
    lo, hi = max(0, peak_idx - half), min(power_t.size, peak_idx + half + 1)

    w = power_t[lo:hi]
    tw = times[lo:hi]
    t_merger = np.sum(tw * w) / np.sum(w)               # energy-weighted peak time
    # spread of the energy in the window, and the error on the MEAN (sqrt of eff. N)
    t_spread = np.sqrt(np.sum(w * (tw - t_merger) ** 2) / np.sum(w))
    n_eff = np.sum(w) ** 2 / np.sum(w ** 2)             # effective number of bins
    t_sigma = t_spread / np.sqrt(n_eff)                 # rough statistical error on the centroid
    return t_merger, t_sigma, power_t


def center_on_peak(times, tf):
    """Shift a time axis so the loudest time column sits at t = 0."""
    return times - times[np.argmax(tf.sum(axis=0))]


# ============================================================
# 1-2. DRAW SOURCE AND SKY PARAMETERS
# ============================================================

# --- Source: random total mass and mass ratio inside NRSur7dq4's valid region ---
total_mass = rng.uniform(60.0, 120.0)                    # Msun; >=60 keeps f_lower=20 Hz valid
mass_ratio = rng.uniform(1.0, 3.0)                       # q = m1/m2 (NRSur7dq4 trained to ~4)
MASS1 = total_mass * mass_ratio / (1.0 + mass_ratio)     # heavier component
MASS2 = total_mass / (1.0 + mass_ratio)                  # lighter component
INCLINATION = np.arccos(rng.uniform(-1.0, 1.0))          # isotropic orientation, 0..pi
COA_PHASE = rng.uniform(0.0, 2.0 * np.pi)                # coalescence phase

# --- Sky location: reject until BOTH detectors respond above MIN_RESPONSE ---
det_h = Detector("H1")
det_l = Detector("L1")
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


# ============================================================
# 3-4. GENERATE, PROJECT, AND TIME-DELAY
# ============================================================

# --- Generate the two polarizations ---
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

# --- Project onto each detector: h(t) = F+ h+ + Fx hx ---
f_plus, f_cross = det_h.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)
f_plus_l, f_cross_l = det_l.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)

# --- Light-travel time delay from Hanford to Livingston for this sky direction ---
dt_HL = det_l.time_delay_from_detector(det_h, RIGHT_ASCENSION, DECLINATION, GPS_TIME)

print(f"H1: F+ = {f_plus:+.4f}, Fx = {f_cross:+.4f}")
print(f"L1: F+ = {f_plus_l:+.4f}, Fx = {f_cross_l:+.4f}")
print(f"H->L time delay: {dt_HL*1e3:+.3f} ms")

# --- Time axes: peak at t = 0 for Hanford; Livingston shifted by the delay ---
peak_index = np.argmax(np.sqrt(h_plus ** 2 + h_cross ** 2))
t = time - time[peak_index]
t_l = t + dt_HL


# ============================================================
# 5. INJECT INTO aLIGO-COLOURED NOISE AT TARGET_SNR
# ============================================================

# --- Detector strains as TimeSeries (so frequency-domain tools work directly) ---
# Both start from the same hp, hc. The physical H->L arrival delay is NOT
# in these samples yet -- I add it explicitly below. Until then the two strains
# are synchronous.
strain_h = f_plus   * hp + f_cross   * hc
strain_l = f_plus_l * hp + f_cross_l * hc

n = len(strain_h)                                    # reference sample count

# --- aLIGO design PSD on this data's frequency grid ---
delta_f = 1.0 / (n * DELTA_T)
psd = aLIGOZeroDetHighPower(n // 2 + 1, delta_f, PSD_LOW_FREQ)

# --- Rescale both strains by the SAME factor so the network SNR = TARGET_SNR ---
# One factor preserves the H/L amplitude ratio, which carries sky information.
# (sigma is time-shift invariant, so it is fine to compute it before the shift.)
sigma_h = sigma(strain_h, psd=psd, low_frequency_cutoff=PSD_LOW_FREQ)
sigma_l = sigma(strain_l, psd=psd, low_frequency_cutoff=PSD_LOW_FREQ)
network_sigma = np.sqrt(sigma_h ** 2 + sigma_l ** 2)
scale = TARGET_SNR / network_sigma
strain_h *= scale
strain_l *= scale

# --- Physically delay the Livingston signal by the light-travel time dt_HL ---
# cyclic_time_shift applies a frequency-domain phase ramp exp(-2*pi*i*f*dt), i.e.
# strain(t - dt), sub-sample accurate. The FFT round-trip can nudge delta_t and
# drop/add a sample, so we rebuild a clean TimeSeries pinned to the original
# DELTA_T, length n, and epoch. Only the signal is shifted -- the noise is drawn
# separately below and must NOT be delayed.
start_l = strain_l.start_time
shifted = strain_l.cyclic_time_shift(-dt_HL).numpy()
if len(shifted) < n:
    shifted = np.pad(shifted, (0, n - len(shifted)))   # pad tail (ringdown ~0 there)
else:
    shifted = shifted[:n]                              # or trim to length
strain_l = TimeSeries(shifted, delta_t=DELTA_T, epoch=start_l)

# --- Independent coloured noise per detector, added to the signal ---
# Independent seeds: the two detectors' noise is physically uncorrelated.
noise_h = noise_from_psd(n, DELTA_T, psd, seed=int(rng.integers(2 ** 31 - 1)))
noise_l = noise_from_psd(n, DELTA_T, psd, seed=int(rng.integers(2 ** 31 - 1)))
noise_h.start_time = strain_h.start_time
noise_l.start_time = strain_l.start_time
data_h = strain_h + noise_h
data_l = strain_l + noise_l

print(f"per-detector SNR: H={sigma_h*scale:.1f}, L={sigma_l*scale:.1f}  "
      f"(network = {TARGET_SNR:.1f})")


# ============================================================
# 6. BAND-PASS AND WHITEN
# ============================================================

bp_h = bandpass(data_h)
bp_l = bandpass(data_l)
clean_bp_h = bandpass(strain_h)                           
clean_bp_l = bandpass(strain_l)
white_h = whiten(bp_h)
white_l = whiten(bp_l)


# ============================================================
# 7. Qp-TRANSFORM AND MERGER TIME
# ============================================================

fg = qp_frequency_grid(BANDPASS_LOW, BANDPASS_HIGH, Q_FIXED, P_BEST)
qp_t, qp_f, qp_tf = qp_transform(white_h.numpy(), FS, fg, Q=Q_FIXED, p=P_BEST)

t_merger_h, t_sigma_h, power_t_h = merger_time_from_energy(qp_t, qp_f, qp_tf)
t_ref = qp_t[np.argmax(qp_tf.sum(axis=0))]             # same reference as the plot
print(f"Hanford merger time = {(t_merger_h - t_ref)*1e3:+.3f} ms "
      f"+/- {t_sigma_h*1e3:.2f} ms")

# --- Livingston: same Qp-transform and energy-peak merger time ---
qp_t_l, qp_f_l, qp_tf_l = qp_transform(white_l.numpy(), FS, fg, Q=Q_FIXED, p=P_BEST)
t_merger_l, t_sigma_l, power_t_l = merger_time_from_energy(qp_t_l, qp_f_l, qp_tf_l)

# --- Recovered H-L delay = difference of the two merger times ---
# Both qp_t and qp_t_l are the same raw time grid, so the raw difference IS the delay.
dt_HL_measured = t_merger_l - t_merger_h
dt_sigma = np.sqrt(t_sigma_h**2 + t_sigma_l**2)   # errors add in quadrature

print(f"\nInjected  H->L delay = {dt_HL*1e3:+.3f} ms")
print(f"Recovered H->L delay = {dt_HL_measured*1e3:+.3f} ms +/- {dt_sigma*1e3:.2f} ms")
print(f"Difference           = {(dt_HL_measured - dt_HL)*1e3:+.3f} ms")

# --- Standard Q-transforms (for the diagnostic plots below) ---
t_qh, f_qh, power_h = white_h.qtransform(delta_t=0.002, logfsteps=300,
                                         qrange=(4, 8), frange=(BANDPASS_LOW, BANDPASS_HIGH))
t_ql, f_ql, power_l = white_l.qtransform(delta_t=0.002, logfsteps=300,
                                         qrange=(4, 8), frange=(BANDPASS_LOW, BANDPASS_HIGH))


# ============================================================
# 8. PLOTS
# ============================================================

# --- Qp-transform map (Hanford) ---
fig_qp, ax_qp = plt.subplots(figsize=(10, 5))
qp_norm = qp_tf * (2.0 / np.median(qp_tf))
ax_qp.pcolormesh(center_on_peak(qp_t, qp_tf), qp_f, qp_norm,
                 shading="auto", cmap="viridis", norm=LogNorm(vmin=2, vmax=qp_norm.max()))
ax_qp.set_yscale("log")
ax_qp.set_xlabel("time (s)")
ax_qp.set_ylabel("freq (Hz)")
ax_qp.set_title(f"Qp-transform: Q={Q_FIXED}, p={P_BEST}")
fig_qp.tight_layout()

# --- Q-transform maps for both detectors ---
fig_q, ax_q = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

m0 = ax_q[0].pcolormesh(t_qh - time[peak_index], f_qh, power_h, shading="auto", cmap="viridis")
ax_q[0].set_yscale("log")
ax_q[0].set_ylabel("frequency (Hz)")
ax_q[0].set_title("Hanford (H1) Q-transform")
fig_q.colorbar(m0, ax=ax_q[0], label="normalized power")

m1 = ax_q[1].pcolormesh(t_ql - time[peak_index] + dt_HL, f_ql, power_l, shading="auto", cmap="viridis")
ax_q[1].set_yscale("log")
ax_q[1].set_ylabel("frequency (Hz)")
ax_q[1].set_xlabel("time relative to Hanford peak (s)")
ax_q[1].set_title("Livingston (L1) Q-transform")
fig_q.colorbar(m1, ax=ax_q[1], label="normalized power")

fig_q.tight_layout()

# --- Whitened strains: clean signal over noisy data ---
fig_s, ax_s = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

wsh = whiten(strain_h).numpy()
wh = white_h.numpy()
mh = min(len(t), len(wsh), len(wh))
ax_s[0].plot(t[:mh], wsh[:mh], color="black", lw=1.5, label="clean")
ax_s[0].plot(t[:mh], wh[:mh], color="tab:purple", alpha=0.6, label="whitened data")
ax_s[0].set_ylabel("strain")
ax_s[0].set_title(f"Hanford (H1)  ($F_+$={f_plus:+.3f}, $F_\\times$={f_cross:+.3f})")
ax_s[0].grid(True, alpha=0.3)
ax_s[0].legend()

wsl = whiten(strain_l).numpy()
wl = white_l.numpy()
ml = min(len(t_l), len(wsl), len(wl))
ax_s[1].plot(t_l[:ml], wsl[:ml], color="black", lw=1.5, label="clean")
ax_s[1].plot(t_l[:ml], wl[:ml], color="tab:green", alpha=0.6, label="whitened data")
ax_s[1].set_ylabel("strain")
ax_s[1].set_xlabel("time relative to Hanford peak (s)")
ax_s[1].set_title(f"Livingston (L1)  ($F_+$={f_plus_l:+.3f}, $F_\\times$={f_cross_l:+.3f})")
ax_s[1].grid(True, alpha=0.3)
ax_s[1].legend()

fig_s.tight_layout()

Path("figures").mkdir(exist_ok=True)
out = Path("figures") / f"{DETECTOR}_strain.png"
fig_s.savefig(out, dpi=200)
print(f"Saved {out}")

plt.show()