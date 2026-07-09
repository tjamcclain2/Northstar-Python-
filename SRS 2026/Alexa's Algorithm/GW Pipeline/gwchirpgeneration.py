import numpy as np
import os
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH   = "/mnt/c/Users/alexa/WnL Docs/Signal Processing/GW Pipeline/data/clean"
SAMPLE_RATE  = 4096       # Hz — LIGO standard
F_LOWER      = 20.0       # Hz — lower frequency cutoff, below this is seismic noise
APPROXIMANT  = "IMRPhenomD"

# -------------------------------------------------------
# PARAMETER GROUPS
# -------------------------------------------------------
# 4 groups x 5 inclination angles = 20 signals total
# Inclination: angle between orbital angular momentum and line of sight
#   0   deg = face-on  (maximum amplitude, both polarizations equal)
#   90  deg = edge-on  (minimum amplitude, only one polarization visible)
#   180 deg = face-on opposite (same amplitude as 0, phase flipped)

INCLINATIONS = [0.0, 45.0, 90.0, 135.0, 180.0]   # degrees

GROUPS = {
    "A_BBH_heavy": {
        "description": "Binary black hole, heavy (30+30 solar masses)",
        "mass1":    30.0,    # solar masses
        "mass2":    30.0,
        "distance": 400.0,   # Mpc
    },
    "B_BNS": {
        "description": "Binary neutron star (1.4+1.4 solar masses)",
        "mass1":    1.4,
        "mass2":    1.4,
        "distance": 100.0,
    },
    "C_BBH_intermediate": {
        "description": "Binary black hole, intermediate (10+10 solar masses)",
        "mass1":    10.0,
        "mass2":    10.0,
        "distance": 200.0,
    },
    "D_mixed": {
        "description": "Mixed NS+BH (30+1.4 solar masses)",
        "mass1":    30.0,
        "mass2":    1.4,
        "distance": 300.0,
    },
}

# -------------------------------------------------------
# GENERATE ONE WAVEFORM
# -------------------------------------------------------
def generate_chirp(mass1, mass2, distance, inclination_deg):

    hp, hc = get_td_waveform(
        approximant  = APPROXIMANT,
        mass1        = mass1,
        mass2        = mass2,
        distance     = distance,
        inclination  = np.deg2rad(inclination_deg),
        f_lower      = F_LOWER,
        delta_t      = 1.0 / SAMPLE_RATE,
    )

    # Combine the two polarizations into a single detector strain
    # h = F+ * hp + Fx * hc
    # For simplicity use a standard antenna response: F+ = 1, Fx = 0
    # This gives the plus polarization only — adequate for algorithm development
    signal = np.array(hp.data, dtype=np.float64)

    # Normalize to peak amplitude of 1.0 so signals are comparable across distances
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    duration = len(signal) / SAMPLE_RATE
    return signal, duration


# -------------------------------------------------------
# GENERATE AND SAVE ALL SIGNALS
# -------------------------------------------------------
os.makedirs(CLEAN_PATH, exist_ok=True)
print(f"Saving clean chirp signals to: {CLEAN_PATH}\n")

metadata = {}
total = 0

for group_name, params in GROUPS.items():
    print(f"Group {group_name}: {params['description']}")
    print(f"  mass1={params['mass1']} M☉  mass2={params['mass2']} M☉  "
          f"distance={params['distance']} Mpc")

    for inc in INCLINATIONS:
        signal_name = f"{group_name}_inc{int(inc):03d}"
        print(f"  Generating: {signal_name}  (inclination={inc}°)", end="")

        signal, duration = generate_chirp(
            mass1          = params["mass1"],
            mass2          = params["mass2"],
            distance       = params["distance"],
            inclination_deg = inc,
        )

        filepath = os.path.join(CLEAN_PATH, f"{signal_name}.npy")
        np.save(filepath, signal)

        metadata[signal_name] = {
            "group":       group_name,
            "mass1":       params["mass1"],
            "mass2":       params["mass2"],
            "distance":    params["distance"],
            "inclination": inc,
            "n_samples":   len(signal),
            "duration_s":  round(duration, 3),
            "sample_rate": SAMPLE_RATE,
        }

        print(f"  →  {len(signal)} samples ({duration:.2f}s)")
        total += 1

    print()

# Save metadata so downstream scripts know group assignments and parameters
np.save(os.path.join(CLEAN_PATH, "metadata.npy"), metadata)

print(f"Done! {total} chirp signals saved.")
print(f"Metadata saved to: {CLEAN_PATH}/metadata.npy")
print("\nSignal durations by group:")
for group_name in GROUPS:
    group_signals = {k: v for k, v in metadata.items() if v["group"] == group_name}
    durations = [v["duration_s"] for v in group_signals.values()]
    print(f"  {group_name}: {min(durations):.2f}s – {max(durations):.2f}s")
