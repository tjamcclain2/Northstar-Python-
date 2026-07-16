"""
plot_nr_chirplet.py
 
    conda activate gw-ligo
    python src/plot_nr_chirplet.py
 
Fit a single analytic "chirplet" (a Gaussian-windowed, linearly-chirping cosine)
to the merger of an NRSur7dq4 gravitational waveform projected onto LIGO-Hanford,
and report how well it reproduces the last few cycles before the peak.
 
A full numerical-relativity surrogate (NRSur7dq4) is expensive to evaluate. This
script tests whether the merger alone can be captured by a cheap closed-form
model with only a handful of parameters (amplitude, frequency, chirp rate, width,
center time, phase). If so, that model is a low-latency stand-in for the merger
portion of the signal. The script fits the model, scores it, and times how much
faster it is to evaluate than the surrogate. Note that this model is not effective
for Northstar as of now, since it cannot deal with noise due to
 
PIPELINE
--------
1. Generate NRSur7dq4 h_plus, h_cross; project to H1:  h = F+ h+ + Fx hx.
2. Form the analytic signal (Hilbert transform) -> instantaneous amplitude
   (the envelope) and instantaneous phase.
3. Merger frequency f_peak = instantaneous frequency at the amplitude peak.
4. FIT REGION = the last FIT_CYCLES cycles before the peak, located by walking
   backwards through the *phase* (so the window self-scales with the signal).
5. Two-stage fit:
     Stage 1  - fit the Gaussian width (lifetime) to the envelope.
     Stage 2  - fit frequency + chirp_rate + phase to the strain, amplitude-
                weighted, seeded by a weighted quadratic fit to the phase.
6. Score with two windows that do NOT feed back into the fit:
     - the last SCORE_CYCLES cycles (the merger the model actually targets),
     - the whole signal (an honest generalisation check; large *by design*).
 
CHIRPLET MODEL
--------------
    Q          = sqrt(ln 2) / lifetime          # NOTE: 'lifetime' is the Half Width at Half Maximum (HWHM) of
                                                #   the envelope, NOT the usual
                                                #   physical quality factor.
    envelope   = exp(-Q^2 dt^2)                 # dt = t - center_time
    phase(t)   = 2*pi*(frequency*dt + 0.5*chirp_rate*dt^2) + phase0
    h(t)       = amplitude * envelope * cos(phase(t))
    inst. freq = frequency + chirp_rate * dt    # linear frequency sweep
 
KEY ASSUMPTIONS
---------------
* Single chirping mode. The model has ONE frequency track. It captures the
  dominant (l=m=2) mode near merger but cannot represent higher modes or the
  full inspiral, whose frequency evolves non-linearly over many cycles.
* Linear chirp. The frequency sweep is approximated as a straight line in time.
  This is only accurate over a SHORT window (a few cycles) near the peak, which
  is exactly why the fit region is small.
* Amplitude is pinned, not fitted. Both fit stages fix the amplitude to the
  measured peak strain (peak_strain). Fitting amplitude and width together is
  degenerate (many (A, width) pairs give a similar curve), so we measure A
  directly and fit only the width. This is a deliberate stability choice, but
  it means the model cannot be used to predict the merger amplitude in a real noise scenario.
* Merger-only target. The model describes the merger/ringdown, not the inspiral.
  Whole-signal error metrics are therefore expected to be large and are reported
  only as an honesty check, never used to drive the fit.
* Fixed source and sky. Masses, spins, distance, inclination and sky position are
  hard-coded constants at the top; change them there to explore other cases.
* No noise. My later pipeline work adds noise and would be better to use as
  a realistic modeling scenario. This script still shows the chirplet fit is an
  effective low-latency merger model, but the fit is not robust to noise currently.
 
DRAWBACKS / THINGS TO WATCH
---------------------------
* Hilbert edge effects. The analytic signal is unreliable within a few samples of
  the array ends. The merger sits in the interior here, so it is fine, but do not
  trust envelope/phase right at the boundaries.
* Fit can fall back. If curve_fit fails to converge, the code keeps the seed
  guess and flags it (width_ok / osc_ok = False). A fallback fit is a warning that
  the region or the bounds may need adjusting, not a silent success.
* Chirp rate can run wild. If the fit region is too short, chirp_rate is poorly
  constrained and may hit its bound. The knob for this is FIT_CYCLES (see below).
* Cycle-windowing needs a clean phase. walk_back_cycles relies on a monotonic
  unwrapped phase; a noisy or multi-mode phase could miscount cycles.
 
TUNING KNOBS (all near the top)
-------------------------------
* FIT_CYCLES     - how many cycles before the peak to fit. If chirp_rate comes
                   back wild, give it a longer lever: raise to 5 or 6.
* SCORE_CYCLES   - how many cycles to SCORE over (independent of the fit).
* WEIGHT_POWER   - residual weighting = envelope**WEIGHT_POWER (2 ~ SNR-like).
* CHIRP_RATE_MAX - bound on |chirp_rate| in the strain fit.
"""

from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from pycbc.waveform import get_td_waveform, td_approximants
from pycbc.detector import Detector
 
 
# ============================================================
# CONFIGURATION
# ============================================================
 
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
# (e.g. the frequency span goes negative), give it a longer lever: raise
# FIT_CYCLES to 4 or 5 while leaving SCORE_CYCLES at 3.
FIT_CYCLES = 3.0
SCORE_CYCLES = 3.0
WEIGHT_POWER = 2.0          # residual weight = envelope**WEIGHT_POWER (2 = power/SNR-like)
CHIRP_RATE_MAX = 5.0e5      # Hz/s bound on the fitted chirp rate
LATENCY_REPEATS = 1000      # times to re-evaluate the model when timing it
 
 
# ============================================================
# MODEL AND FIT FUNCTIONS
# ============================================================
 
def chirplet(t, amplitude, frequency, chirp_rate, lifetime, center_time, phase):
    """The chirplet model: a Gaussian-windowed cosine with a linear frequency sweep.
 
    This is the closed-form stand-in for the merger. Evaluate it on any time array
    to get a model strain. It is used both as the thing being fitted (via the
    osc_model wrapper below) and to generate the final model_strain for plotting
    and scoring.
 
    Parameters
    ----------
    t : ndarray
        Times at which to evaluate the model (seconds), same clock as the data.
    amplitude : float
        Peak strain of the model. Here it is pinned to the measured peak, not fitted.
    frequency : float
        Instantaneous frequency at center_time (Hz).
    chirp_rate : float
        Linear rate of change of frequency (Hz/s); positive = frequency rising.
    lifetime : float
        Half-width at half-maximum of the Gaussian envelope (seconds). Larger =
        wider, longer-lived merger. Converted to the Gaussian factor Q below.
    center_time : float
        Time of the envelope peak (seconds); dt is measured relative to this.
    phase : float
        Phase offset of the cosine at center_time (radians).
 
    Returns
    -------
    ndarray
        Model strain at each time in t.
 
    Notes
    -----
    Q = sqrt(ln 2)/lifetime makes the envelope fall to half its peak when
    |dt| = lifetime, i.e. 'lifetime' is the half-width at half-maximum. This is a
    convenience definition and is NOT the physical quality factor of the mode.
    """
    q = np.sqrt(np.log(2.0)) / lifetime                       # Gaussian factor from HWHM
    dt = t - center_time                                      # time relative to the peak
    envelope = np.exp(-(q ** 2) * dt ** 2)                    # Gaussian amplitude envelope
    inst_phase = 2.0 * np.pi * (frequency * dt + 0.5 * chirp_rate * dt ** 2) + phase
    return amplitude * envelope * np.cos(inst_phase)          # windowed chirping cosine
 
 
def walk_back_cycles(n_cycles):
    """Return the sample index that lies n_cycles of phase BEFORE the amplitude peak.
 
    This is how the fit and score windows are defined: instead of a fixed number of
    milliseconds, we step backwards from the peak until we have covered n_cycles of
    the waveform's own phase. That makes every window self-scaling -- a low-frequency
    system automatically gets a longer window in time than a high-frequency one, both
    covering the same number of oscillations.
 
    Parameters
    ----------
    n_cycles : float
        Number of oscillation cycles to walk back from the peak.
 
    Returns
    -------
    int
        Index into the sample arrays marking the start of that window.
 
    Notes
    -----
    Relies on module-level globals produced earlier in the script:
    peak_index (index of the amplitude peak), analytic_phase (unwrapped phase of
    the analytic signal), and phase_at_peak (that phase at the peak). It assumes the
    phase is monotonic and clean; a noisy or multi-mode phase could miscount cycles.
    One cycle corresponds to a phase change of 2*pi.
    """
    j = peak_index
    # step left until the accumulated phase back from the peak reaches n_cycles*2pi
    while j > 0 and (phase_at_peak - analytic_phase[j]) < n_cycles * 2.0 * np.pi:
        j -= 1
    return j
 
 
# NOTE: gaussian_envelope and osc_model are defined further below, AFTER the
# quantities they pin (peak_strain, peak_center, fit_lifetime) have been measured.
# They are thin wrappers that expose only the parameters each fit stage varies,
# which is what scipy.optimize.curve_fit expects. For a noise scenario, one could either
# use a single-stage fit with all six parameters free, or use a more sophisticated
# optimizer that pins certain parameters (such as amplitude, lifetime, and center_time) 
# while varying the others. The two-stage fit here is a deliberate stability choice for 
# this noiseless test, but it is not the only way to do it.
 
 
# ============================================================
# GENERATE + PROJECT
# ============================================================
 
Path("figures").mkdir(exist_ok=True)

# Guard: NRSur7dq4 needs the lalsimulation surrogate data package installed.
if APPROXIMANT not in td_approximants():
    raise SystemExit(f"ERROR: {APPROXIMANT} unavailable. Try: "
                     f"mamba install -n gw-ligo -c conda-forge lalsimulation-data -y")
 
print(f"Generating {APPROXIMANT} ...")
t_gen = perf_counter()
hp, hc = get_td_waveform(                                    # generate the surrogate via NRSur7dq4
    approximant=APPROXIMANT, mass1=MASS1, mass2=MASS2,
    spin1x=SPIN1X, spin1y=SPIN1Y, spin1z=SPIN1Z,
    spin2x=SPIN2X, spin2y=SPIN2Y, spin2z=SPIN2Z,
    distance=DISTANCE, inclination=INCLINATION, coa_phase=COALESCENCE_PHASE,
    delta_t=DELTA_T, f_lower=F_LOWER, f_ref=F_REF)
generation_time = perf_counter() - t_gen                     # cost of the full surrogate
 
# Pull numpy arrays out of the pycbc TimeSeries and project onto the detector.
time_values = hp.sample_times.numpy()
h_plus, h_cross = hp.numpy(), hc.numpy()
detector = Detector(DETECTOR)
f_plus, f_cross = detector.antenna_pattern(RIGHT_ASCENSION, DECLINATION, POLARIZATION, GPS_TIME)
detector_strain = f_plus * h_plus + f_cross * h_cross        # h = F+ h+ + Fx hx
print(f"{DETECTOR}: F+={f_plus:+.4f}  Fx={f_cross:+.4f}  |F|={np.hypot(f_plus, f_cross):.4f}")
 
# Peak of the polarization amplitude marks the merger; use it as t = 0.
amplitude = np.sqrt(h_plus ** 2 + h_cross ** 2)
peak_index = int(np.argmax(amplitude))
time_shifted = time_values - time_values[peak_index]         # time relative to the peak
signal = detector_strain
n_samples = signal.size
 
 
# ============================================================
# ANALYTIC SIGNAL (envelope + instantaneous frequency)
# ============================================================
 
# The Hilbert transform builds the analytic signal, whose magnitude is the
# instantaneous amplitude (envelope) and whose phase derivative is the
# instantaneous frequency. These drive both the windowing and the fit seeds.
analytic = hilbert(signal)
analytic_env = np.abs(analytic)                              # instantaneous amplitude
analytic_phase = np.unwrap(np.angle(analytic))              # continuous (unwrapped) phase
inst_freq = np.gradient(analytic_phase, time_shifted) / (2.0 * np.pi)   # d(phase)/dt / 2pi
 
# Reference quantities measured at the peak (used to pin the model and seed the fit).
peak_strain = float(np.max(np.abs(signal)))                 # pinned model amplitude
peak_center = float(time_shifted[peak_index])               # pinned model center time (~0)
peak_env = analytic_env[peak_index]
f_peak = abs(inst_freq[peak_index])                         # merger frequency estimate
amp_guess = peak_strain                                     # NRMSE normalization
span = time_shifted[-1] - time_shifted[0]                   # full signal duration
nyquist = 0.5 / DELTA_T                                     # frequency upper bound for the fit
phase_at_peak = analytic_phase[peak_index]                  # used by walk_back_cycles
print(f"[check] merger frequency f_peak = {f_peak:.1f} Hz   (expect ~150-250 Hz)")
 
 
# ============================================================
# FIT REGION: the last FIT_CYCLES cycles before the peak
# ============================================================
 
i_fit = walk_back_cycles(FIT_CYCLES)
region = slice(i_fit, peak_index + 1)                       # samples in the fit window
t_region = time_shifted[region]
env_region = analytic_env[region]
dt_region = t_region - peak_center
 
# Residual weights: emphasise high-amplitude samples (near the peak) so the fit
# cares most about the loudest, best-measured part of the merger. 'sigma' here is
# the per-point uncertainty curve_fit expects (small where weight is large), NOT
# a signal-to-noise ratio.
weights = env_region ** WEIGHT_POWER
sigma = 1.0 / (weights / weights.max() + 1e-3)
 
 
# ---- Stage 1: Gaussian width (lifetime) from the envelope ----
 
# Start the width at a fraction of the fit-region span, so it begins at merger scale.
lifetime0 = max((peak_center - t_region[0]) / 2.0, 2.0 * DELTA_T)
 
 
def gaussian_envelope(t, lifetime):
    """Stage-1 model: the chirplet's Gaussian ENVELOPE only, amplitude pinned.
 
    Fitted against the measured envelope to determine the width (lifetime). The
    amplitude is fixed to peak_strain (measured, not fitted) to avoid the
    amplitude/width degeneracy, so lifetime is the single free parameter.
 
    Parameters
    ----------
    t : ndarray
        Times (seconds).
    lifetime : float
        Envelope HWHM (seconds) -- the parameter curve_fit varies.
 
    Returns
    -------
    ndarray
        Envelope values peak_strain * exp(-Q^2 (t-center)^2), Q = sqrt(ln2)/lifetime.
    """
    q = np.sqrt(np.log(2.0)) / lifetime
    return peak_strain * np.exp(-(q ** 2) * (t - peak_center) ** 2)
 
 
try:
    # Fit the width; bound it between one sample and the whole span. If it fails,
    # keep the seed and flag width_ok = False rather than crashing.
    (fit_lifetime,), _ = curve_fit(gaussian_envelope, t_region, env_region,
                               p0=[lifetime0], sigma=sigma, bounds=([DELTA_T], [span]),
                               maxfev=20000)
    width_ok = True
except RuntimeError:
    fit_lifetime, width_ok = lifetime0, False
 
 
# ---- Stage 2a: weighted quadratic phase fit -> robust freq/chirp/phase seeds ----
 
# A cosine's phase is 2*pi*(f*dt + 0.5*chirp*dt^2) + phase0, i.e. quadratic in dt.
# Fitting a weighted parabola to the measured phase gives clean starting guesses
# for the (harder) non-linear strain fit that follows.
c2, c1, c0 = np.polyfit(dt_region, analytic_phase[region], 2, w=weights)
freq0 = abs(c1) / (2.0 * np.pi)                             # linear term  -> frequency
chirp0 = c2 / np.pi                                         # quadratic term -> chirp rate
phase0 = (c0 + np.pi) % (2.0 * np.pi) - np.pi               # constant term -> phase, wrapped to (-pi, pi]
 
 
# ---- Stage 2b: amplitude-weighted strain fit ----
 
def osc_model(t, frequency, chirp_rate, phase):
    """Stage-2 model: the full chirplet with amplitude and width already pinned.
 
    Exposes only the three parameters this stage fits (frequency, chirp_rate,
    phase). amplitude is fixed to peak_strain, lifetime to the Stage-1 result
    (fit_lifetime), and center_time to peak_center -- so curve_fit varies exactly
    the phase-track parameters and nothing that is already measured.
 
    Parameters
    ----------
    t : ndarray
        Times (seconds).
    frequency, chirp_rate, phase : float
        The three fitted parameters (see chirplet docstring).
 
    Returns
    -------
    ndarray
        Model strain over t.
    """
    return chirplet(t, peak_strain, frequency, chirp_rate, fit_lifetime, peak_center, phase)
 
 
t_fit = perf_counter()
try:
    # Non-linear least squares on the strain, seeded by the quadratic-phase guesses
    # and weighted toward the loud samples. Bounds keep the parameters physical.
    (fit_frequency, fit_chirp, fit_phase), _ = curve_fit(
        osc_model, t_region, signal[region], p0=[freq0, chirp0, phase0], sigma=sigma,
        bounds=([1.0, -CHIRP_RATE_MAX, -2.0 * np.pi], [nyquist, CHIRP_RATE_MAX, 2.0 * np.pi]),
        maxfev=20000)
    osc_ok = True
except RuntimeError:
    fit_frequency, fit_chirp, fit_phase, osc_ok = freq0, chirp0, phase0, False
fit_time = perf_counter() - t_fit
 
# Assemble the full six-parameter vector for the final model.
popt = np.array([peak_strain, fit_frequency, fit_chirp, fit_lifetime, peak_center, fit_phase])
 
 
# ============================================================
# LATENCY: cost of evaluating the model vs generating the surrogate
# ============================================================
 
# Evaluate the model many times and keep the fastest run (best-case latency, least
# polluted by scheduling jitter). This is the number compared against the surrogate
# generation time to quote a speed-up.
best_latency = np.inf
model_strain = None
for _ in range(LATENCY_REPEATS):
    t0 = perf_counter()
    model_strain = chirplet(time_shifted, *popt)
    best_latency = min(best_latency, perf_counter() - t0)
chirplet_latency = best_latency
 
# Model envelope (for plotting) and the fit residual.
q_fit = np.sqrt(np.log(2.0)) / fit_lifetime
model_envelope = peak_strain * np.exp(-(q_fit ** 2) * (time_shifted - peak_center) ** 2)
residual = signal - model_strain               #Used to graph how well the model fits the data, and to compute NRMSE.
 
 
# ============================================================
# SCORING (two windows, neither feeds back into the fit)
# ============================================================
 
# Window A: the last SCORE_CYCLES cycles -- the merger the model targets.
# Window B: the whole signal -- an honest generalisation check (large by design,
#           because the model is not meant to fit the long inspiral).
i_score = walk_back_cycles(SCORE_CYCLES)
score_mask_cycles = np.zeros(n_samples, dtype=bool)
score_mask_cycles[i_score:peak_index + 1] = True
score_mask_full = np.ones(n_samples, dtype=bool)
 
# Normalized RMS errors (residual RMS divided by the peak amplitude).
nrmse_cycles = float(np.sqrt(np.mean(residual[score_mask_cycles] ** 2))) / amp_guess
nrmse_full = float(np.sqrt(np.mean(residual[score_mask_full] ** 2))) / amp_guess
nrmse_weighted = float(np.sqrt(np.sum(weights * residual[region] ** 2) / np.sum(weights))) / amp_guess
 
# Extra diagnostics over the cycle window.
a = signal[score_mask_cycles]
b = model_strain[score_mask_cycles]
overlap = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))   # phase alignment, 1 = perfect
env_nrmse = float(np.sqrt(np.mean(
    (analytic_env[score_mask_cycles] - model_envelope[score_mask_cycles]) ** 2))) / amp_guess
f_lo = fit_frequency + fit_chirp * (t_region[0] - peak_center)   # frequency at the start of the fit window
 
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
 
 
# ============================================================
# PLOT
# ============================================================
 
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
 
# Panel 0-1: the raw polarizations.
axes[0].plot(time_shifted, h_plus, color="tab:blue"); axes[0].set_ylabel(r"$h_+$")
axes[0].set_title(f"{APPROXIMANT} on {DETECTOR}: cycle-windowed chirplet fit")
axes[1].plot(time_shifted, h_cross, color="tab:orange"); axes[1].set_ylabel(r"$h_\times$")
 
# Panel 2: detector strain, the fitted chirplet, its envelope, and the two windows.
axes[2].plot(time_shifted, detector_strain, color="tab:purple", label=r"$h_{\rm det}$")
axes[2].plot(time_shifted, model_strain, "--", color="k", label="chirplet")
axes[2].plot(time_shifted, model_envelope, ":", color="tab:green", lw=1)
axes[2].plot(time_shifted, -model_envelope, ":", color="tab:green", lw=1, label="envelope")
axes[2].axvspan(t_region[0], peak_center, alpha=0.06, color="tab:blue", label="fit region")
axes[2].axvspan(time_shifted[i_score], peak_center, alpha=0.15, color="tab:green", label="score (cycles)")
axes[2].set_ylabel(r"$h_{\rm det}$"); axes[2].legend(loc="upper left", fontsize=8)
 
# Panel 3: fit residual, with the score window marked.
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