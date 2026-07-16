"""
NOTE: This is an older version of my chirplet modeling and has fewer comments and generally
worse design compared to chirplet_pipeline.py. I reccomend starting there.

conda activate gw-ligo
python src/plot_nr_chirplet.py

NRSur7dq4 -> LIGO-Hanford projection -> CHIRPLET fit, with all windowing driven
by CYCLES of the waveform (self-scaling, no hardcoded milliseconds).

Pipeline
--------
1. Generate NRSur7dq4 h_plus, h_cross; project to H1: h = F+ h+ + Fx hx.
2. Analytic signal (Hilbert) -> instantaneous amplitude (envelope) and phase.
3. Merger frequency f_peak = instantaneous frequency at the amplitude peak.
4. FIT REGION  = last FIT_CYCLES cycles before the peak (found from the phase).
5. Stage 1: fit the Gaussian width to the envelope over the fit region.
   Stage 2: fit frequency + chirp_rate + phase to the strain over the fit region
            (amplitude-weighted), seeded by a weighted quadratic phase fit.
6. Score with TWO windows that feed back into nothing:
     - last SCORE_CYCLES cycles (the merger the model targets)
     - the whole signal (honest generalisation check; will be large by design)

Chirplet model
--------------
    Q          = sqrt(ln 2) / lifetime
    envelope   = exp(-Q^2 dt^2)                        dt = t - center
    phase(t)   = 2*pi*(frequency*dt + 0.5*chirp_rate*dt^2) + phase0
    h(t)       = amplitude * envelope * cos(phase(t))
    inst. freq = frequency + chirp_rate * dt
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from pycbc.waveform import get_td_waveform, td_approximants
from pycbc.detector import Detector

# ---- Source / waveform ----
APPROXIMANT = "NRSur7dq4"
MASS1, MASS2 = 45.0, 30.0
DISTANCE, INCLINATION, COALESCENCE_PHASE = 400.0, 0.4, 0.0
SPIN1X, SPIN1Y, SPIN1Z = 0.0, 0.0, 0.2
SPIN2X, SPIN2Y, SPIN2Z = 0.0, 0.0, -0.1
DELTA_T = 1.0 / 4096.0
F_LOWER = 20.0
F_REF = F_LOWER

# ---- Detector / sky location ----
DETECTOR = "H1"
RIGHT_ASCENSION, DECLINATION, POLARIZATION = 1.95, -1.27, 0.0
GPS_TIME = 1126259462.0

# ---- Cycle-based windowing ----
# Fit the last FIT_CYCLES cycles before the peak. If chirp_rate comes back wild
# (e.g. the span goes negative), give it a longer lever: raise FIT_CYCLES to ~6
# while leaving SCORE_CYCLES at 4.
FIT_CYCLES = 3.0
SCORE_CYCLES = 3.0
WEIGHT_POWER = 2.0          # residual weight = envelope**WEIGHT_POWER (2 = power/SNR-like)
CHIRP_RATE_MAX = 5.0e5      # Hz/s bound
LATENCY_REPEATS = 1000


def chirplet(t, amplitude, frequency, chirp_rate, lifetime, center_time, phase):
    q = np.sqrt(np.log(2.0)) / lifetime
    dt = t - center_time
    envelope = np.exp(-(q ** 2) * dt ** 2)
    inst_phase = 2.0 * np.pi * (frequency * dt + 0.5 * chirp_rate * dt ** 2) + phase
    return amplitude * envelope * np.cos(inst_phase)


# ---- Generate + project ----
Path("figures").mkdir(exist_ok=True)
if APPROXIMANT not in td_approximants():
    raise SystemExit(f"ERROR: {APPROXIMANT} unavailable. Try: "
                     f"mamba install -n gw-ligo -c conda-forge lalsimulation-data -y")

print(f"Generating {APPROXIMANT} ...")
t_gen = perf_counter()
hp, hc = get_td_waveform(
    approximant=APPROXIMANT, mass1=MASS1, mass2=MASS2,
    spin1x=SPIN1X, spin1y=SPIN1Y, spin1z=SPIN1Z,
    spin2x=SPIN2X, spin2y=SPIN2Y, spin2z=SPIN2Z,
    distance=DISTANCE, inclination=INCLINATION, coa_phase=COALESCENCE_PHASE,
    delta_t=DELTA_T, f_lower=F_LOWER, f_ref=F_REF)
generation_time = perf_counter() - t_gen

time_values = hp.sample_times.numpy()
h_plus, h_cross = hp.numpy(), hc.numpy()
detector = Detector(DETECTOR)
f_plus, f_cross = detector.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)
detector_strain = f_plus * h_plus + f_cross * h_cross
print(f"{DETECTOR}: F+={f_plus:+.4f}  Fx={f_cross:+.4f}  |F|={np.hypot(f_plus, f_cross):.4f}")

amplitude = np.sqrt(h_plus ** 2 + h_cross ** 2)
peak_index = int(np.argmax(amplitude))
time_shifted = time_values - time_values[peak_index]
signal = detector_strain
n_samples = signal.size

# ---- Analytic signal ----
analytic = hilbert(signal)
analytic_env = np.abs(analytic)
analytic_phase = np.unwrap(np.angle(analytic))
inst_freq = np.gradient(analytic_phase, time_shifted) / (2.0 * np.pi)

peak_strain = float(np.max(np.abs(signal)))
peak_center = float(time_shifted[peak_index])
peak_env = analytic_env[peak_index]
f_peak = abs(inst_freq[peak_index])
amp_guess = peak_strain
span = time_shifted[-1] - time_shifted[0]
nyquist = 0.5 / DELTA_T
phase_at_peak = analytic_phase[peak_index]
print(f"[check] merger frequency f_peak = {f_peak:.1f} Hz   (expect ~150-250 Hz)")


def walk_back_cycles(n_cycles):
    """Index of the sample n_cycles of phase before the peak."""
    j = peak_index
    while j > 0 and (phase_at_peak - analytic_phase[j]) < n_cycles * 2.0 * np.pi:
        j -= 1
    return j


# ---- Fit region: last FIT_CYCLES cycles ----
i_fit = walk_back_cycles(FIT_CYCLES)
region = slice(i_fit, peak_index + 1)
t_region = time_shifted[region]
env_region = analytic_env[region]
dt_region = t_region - peak_center
weights = env_region ** WEIGHT_POWER
sigma = 1.0 / (weights / weights.max() + 1e-3)

# ---- Stage 1: Gaussian width from the envelope over the fit region ----
# width guess = a fraction of the fit-region span, so it starts at merger scale
lifetime0 = max((peak_center - t_region[0]) / 2.0, 2.0 * DELTA_T)

def gaussian_envelope(t, lifetime):
    q = np.sqrt(np.log(2.0)) / lifetime
    return peak_strain * np.exp(-(q ** 2) * (t - peak_center) ** 2)

try:
    (fit_lifetime,), _ = curve_fit(gaussian_envelope, t_region, env_region,
                               p0=[lifetime0], sigma=sigma, bounds=([DELTA_T], [span]),
                               maxfev=20000)
    width_ok = True
except RuntimeError:
    fit_lifetime, width_ok = lifetime0, False

# ---- Stage 2a: weighted quadratic phase fit -> robust freq/chirp/phase guess ----
c2, c1, c0 = np.polyfit(dt_region, analytic_phase[region], 2, w=weights)
freq0 = abs(c1) / (2.0 * np.pi)
chirp0 = c2 / np.pi
phase0 = (c0 + np.pi) % (2.0 * np.pi) - np.pi

# ---- Stage 2b: amplitude-weighted strain fit ----
def osc_model(t, frequency, chirp_rate, phase):
    return chirplet(t, peak_strain, frequency, chirp_rate, fit_lifetime, peak_center, phase)

t_fit = perf_counter()
try:
    (fit_frequency, fit_chirp, fit_phase), _ = curve_fit(
        osc_model, t_region, signal[region], p0=[freq0, chirp0, phase0], sigma=sigma,
        bounds=([1.0, -CHIRP_RATE_MAX, -2.0 * np.pi], [nyquist, CHIRP_RATE_MAX, 2.0 * np.pi]),
        maxfev=20000)
    osc_ok = True
except RuntimeError:
    fit_frequency, fit_chirp, fit_phase, osc_ok = freq0, chirp0, phase0, False
fit_time = perf_counter() - t_fit

popt = np.array([peak_strain, fit_frequency, fit_chirp, fit_lifetime, peak_center, fit_phase])

# ---- Latency ----
best_latency = np.inf
model_strain = None
for _ in range(LATENCY_REPEATS):
    t0 = perf_counter()
    model_strain = chirplet(time_shifted, *popt)
    best_latency = min(best_latency, perf_counter() - t0)
chirplet_latency = best_latency

q_fit = np.sqrt(np.log(2.0)) / fit_lifetime
model_envelope = peak_strain * np.exp(-(q_fit ** 2) * (time_shifted - peak_center) ** 2)
residual = signal - model_strain

# ---- Two score windows (independent of the fit) ----
i_score = walk_back_cycles(SCORE_CYCLES)
score_mask_cycles = np.zeros(n_samples, dtype=bool)
score_mask_cycles[i_score:peak_index + 1] = True
score_mask_full = np.ones(n_samples, dtype=bool)

nrmse_cycles = float(np.sqrt(np.mean(residual[score_mask_cycles] ** 2))) / amp_guess
nrmse_full = float(np.sqrt(np.mean(residual[score_mask_full] ** 2))) / amp_guess
nrmse_weighted = float(np.sqrt(np.sum(weights * residual[region] ** 2) / np.sum(weights))) / amp_guess

# ---- Diagnostics ----
a = signal[score_mask_cycles]
b = model_strain[score_mask_cycles]
overlap = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
env_nrmse = float(np.sqrt(np.mean(
    (analytic_env[score_mask_cycles] - model_envelope[score_mask_cycles]) ** 2))) / amp_guess
f_lo = fit_frequency + fit_chirp * (t_region[0] - peak_center)

print("\n--- DIAGNOSTICS ---")
print(f"width_ok={width_ok}  osc_ok={osc_ok}")
print(f"f_peak            : {f_peak:.1f} Hz")
print(f"fit region        : [{t_region[0]*1e3:.2f}, 0] ms   ({env_region.size} samples, {FIT_CYCLES:.0f} cycles)")
print(f"score (cycles)    : [{time_shifted[i_score]*1e3:.2f}, 0] ms   ({int(score_mask_cycles.sum())} samples, {SCORE_CYCLES:.0f} cycles)")
print(f"fit_lifetime      : {fit_lifetime*1e3:.2f} ms")
print(f"frequency / chirp : {fit_frequency:.1f} Hz  /  {fit_chirp:.0f} Hz/s")
print(f"freq span in fit  : {f_lo:.0f} -> {fit_frequency:.0f} Hz")
print(f"overlap (cycles)  : {overlap:+.3f}   (1 = perfect phase alignment)")
print(f"envelope-only NRMSE (cycles) : {env_nrmse:.3f}")
print("--- END ---\n")

print("Chirplet fit" + ("" if (width_ok and osc_ok) else "  [a stage fell back]"))
print(f"  NRMSE last {SCORE_CYCLES:.0f} cycles : {nrmse_cycles:.3f}")
print(f"  NRMSE whole signal   : {nrmse_full:.3f}   (large by design: merger model, not inspiral)")
print(f"  NRMSE amp-weighted   : {nrmse_weighted:.3f}")
print(f"Latency: NRSur {generation_time*1e3:.2f} ms | chirplet {chirplet_latency*1e6:.2f} us"
      + (f" | speed-up {generation_time/chirplet_latency:,.0f}x" if chirplet_latency > 0 else ""))

# ---- Plot ----
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
axes[0].plot(time_shifted, h_plus, color="tab:blue"); axes[0].set_ylabel(r"$h_+$")
axes[0].set_title(f"{APPROXIMANT} on {DETECTOR}: cycle-windowed chirplet fit")
axes[1].plot(time_shifted, h_cross, color="tab:orange"); axes[1].set_ylabel(r"$h_\times$")
axes[2].plot(time_shifted, detector_strain, color="tab:purple", label=r"$h_{\rm det}$")
axes[2].plot(time_shifted, model_strain, "--", color="k", label="chirplet")
axes[2].plot(time_shifted, model_envelope, ":", color="tab:green", lw=1)
axes[2].plot(time_shifted, -model_envelope, ":", color="tab:green", lw=1, label="envelope")
axes[2].axvspan(t_region[0], peak_center, alpha=0.06, color="tab:blue", label="fit region")
axes[2].axvspan(time_shifted[i_score], peak_center, alpha=0.15, color="tab:green", label="score (cycles)")
axes[2].set_ylabel(r"$h_{\rm det}$"); axes[2].legend(loc="upper left", fontsize=8)
axes[3].plot(time_shifted, residual, color="tab:red")
axes[3].axvspan(time_shifted[i_score], peak_center, alpha=0.15, color="tab:green")
axes[3].set_xlabel("Time relative to peak (s)"); axes[3].set_ylabel("residual")
for ax in axes:
    ax.axvline(0, ls=":", lw=1, color="k"); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = Path("figures") / f"{APPROXIMANT}_{DETECTOR}_chirplet_cycles.png"
fig.savefig(out, dpi=200)
print(f"Saved plot to: {out}")
plt.show()
