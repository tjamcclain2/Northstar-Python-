"""
generate_strain.py

Minimal pipeline: generate an NRSur7dq4 waveform with fixed parameters, project
it onto Hanford and Livingston for a fixed sky angle to get the detector strain, add noise, and plot.

    conda activate gw-ligo
    python src/generate_strain.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma

# ---- Random number generator ----
SEED = None                      # set to an int for reproducible runs
rng = np.random.default_rng(SEED)

# ---- Source parameters ----
APPROXIMANT = "NRSur7dq4"
SPIN1Z, SPIN2Z = 0.2, -0.1
DISTANCE = 400.0        # Mpc
DELTA_T = 1.0 / 4096.0  # s
F_LOWER = 20.0          # Hz
TARGET_SNR = 20.0      # tunable network matched-filter SNR
PSD_LOW_FREQ = 20.0    # Hz; lower edge for SNR/noise, matches F_LOWER
# --- masses: draw total mass and mass ratio inside NRSur7dq4's valid region ---
total_mass = rng.uniform(60.0, 120.0)   # Msun; >=60 keeps f_lower=20 Hz valid
mass_ratio = rng.uniform(1.0, 4.0)      # q = m1/m2, NRSur7dq4 trained up to ~4
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
MIN_RESPONSE = 0.4   # require |F| = sqrt(F+^2 + Fx^2) >= this in each detector

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

print(f"per-detector SNR: H={sigma_h*scale:.1f}, L={sigma_l*scale:.1f}  "
      f"(network = {TARGET_SNR:.1f})")

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t, data_h.numpy(), color="tab:purple", alpha=0.5, label="signal + noise")
axes[0].plot(t, strain_h.numpy(), color="black", lw=1.5, label="clean signal")
axes[0].legend()
axes[0].set_ylabel("strain")
axes[0].set_title(f"Hanford (H1)  ($F_+$={f_plus:+.3f}, $F_\\times$={f_cross:+.3f})")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_l, data_l.numpy(), color="tab:green", alpha=0.5, label="signal + noise")
axes[1].plot(t_l, strain_l.numpy(), color="black", lw=1.5, label="clean signal")
axes[1].legend()
axes[1].set_ylabel("strain")
axes[1].set_xlabel("time relative to Hanford peak (s)")
axes[1].set_title(f"Livingston (L1)  ($F_+$={f_plus_l:+.3f}, $F_\\times$={f_cross_l:+.3f})")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
out = Path("figures") / f"{DETECTOR}_strain.png"
fig.savefig(out, dpi=200)
print(f"Saved {out}")
plt.show()