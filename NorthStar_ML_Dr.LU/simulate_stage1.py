#!/usr/bin/env python3
"""
simulate_stage1.py
------------------
Generate synthetic two-detector (Hanford & Livingston) gravitational-wave-like signals
for Stage 1 of the ML weighting project.

What it does
------------
- Samples 10,000 (configurable) events with randomized sky location, polarization,
  inclination, carrier frequency, and target network SNR.
- Builds simple narrowband waveforms (Gaussian-windowed sin/cos pair for h+, h×).
- Computes detector antenna patterns F+, F× using detector arm tensors.
- Applies inter-site geometric time delay.
- Injects into Gaussian noise and scales to the requested network SNR.
- Saves per-event .npz files with H/L strains and a manifest CSV with metadata.
- (Optional) Sweeps a set of weighting exponents n by calling a user-provided hook
  function `run_localization(event_npz_path, n)` in a separate module.

Usage
-----
Basic generation only:
    python simulate_stage1.py --out data/injections --num 10000

Generate a small test set:
    python simulate_stage1.py --out data/injections_test --num 50 --seed 123 --fs 4096 --T 1.0

Sweep n via your localization hook:
    python simulate_stage1.py --out data/injections --num 10000 --sweep-n \
        --hook-module northstar_hook --alphabeta 1.0 1.0

The hook module (northstar_hook.py) must be importable (on PYTHONPATH) and define:
    def run_localization(event_npz_path: str, n_value: float) -> dict:
        \"\"\"Return a dict with keys 'deltaF_rms' and 'deltatau_rms'.\"\"\"

Outputs
-------
- out_dir/sample_000001.npz (per-event signals & metadata)
- out_dir/manifest.csv (one row per event with truth metadata)
- out_dir/results.csv (only if --sweep-n; one row per event per n with errors)
- out_dir/labels.csv  (only if --sweep-n; best n* per event by weighted sum)

Notes
-----
- This is a minimal physics-faithful generator aimed at ML data creation and algorithm testing.
- Replace detector arm vectors with your exact values if you have them.

Goal: Given an injected event, predict the optimal weighting exponent n* that 
minimizes sky-localization error for NorthStar.

"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq


C = 299_792_458.0  # speed of light, m/s


# -----------------------------------------------------------------------------
# Detector geometry (ECEF positions in meters + approximate arm unit vectors)
# Replace with exact values if available.
# -----------------------------------------------------------------------------
r_H = np.array([-2161414.926, -3834698.371,  4600350.226], dtype=float)
r_L = np.array([  -74276.044, -5496283.719,  3224257.017], dtype=float)

# Approximate arm unit vectors (ensure they are orthonormal per detector)
u_H = np.array([ 0.891,  0.323,  0.319], dtype=float)
v_H = np.array([-0.453,  0.836,  0.311], dtype=float)
u_H /= np.linalg.norm(u_H); v_H -= np.dot(u_H, v_H) * u_H; v_H /= np.linalg.norm(v_H)

u_L = np.array([ 0.954, -0.141,  0.266], dtype=float)
v_L = np.array([ 0.297,  0.487, -0.822], dtype=float)
u_L /= np.linalg.norm(u_L); v_L -= np.dot(u_L, v_L) * u_L; v_L /= np.linalg.norm(v_L)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class EventMeta:
    idx: int
    seed: int
    fs: float
    T: float
    f: float
    snr: float
    theta: float
    phi: float
    psi: float
    iota: float
    tau: float
    FH_plus: float
    FH_cross: float
    FL_plus: float
    FL_cross: float


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def unit_vec_from_sky(theta: float, phi: float) -> np.ndarray:
    """Return source direction unit vector n_hat given colatitude theta and longitude phi."""
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ], dtype=float)


def polarization_basis(n_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return k (prop. dir), p and q polarization basis vectors s.t. p,q ⟂ k; right-handed."""
    k = n_hat / (np.linalg.norm(n_hat) + 1e-15)
    # pick a helper axis not parallel to k
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, tmp)) > 0.99:
        tmp = np.array([0.0, 1.0, 0.0])
    p = np.cross(k, tmp); p /= (np.linalg.norm(p) + 1e-15)
    q = np.cross(k, p)
    return k, p, q


def antenna_pattern(n_hat: np.ndarray, psi: float, u_hat: np.ndarray, v_hat: np.ndarray) -> Tuple[float, float]:
    """Compute (F_plus, F_cross) for an L-shaped detector with arms u_hat and v_hat."""
    d = 0.5 * (np.outer(u_hat, u_hat) - np.outer(v_hat, v_hat))  # detector tensor
    _, p, q = polarization_basis(n_hat)
    # base polarization tensors
    e_plus  = np.outer(p, p) - np.outer(q, q)
    e_cross = np.outer(p, q) + np.outer(q, p)
    # rotate by polarization angle psi
    c2 = math.cos(2.0 * psi)
    s2 = math.sin(2.0 * psi)
    e_plus_r  = c2 * e_plus - s2 * e_cross
    e_cross_r = s2 * e_plus + c2 * e_cross
    Fp = float(np.tensordot(d, e_plus_r, axes=2))
    Fx = float(np.tensordot(d, e_cross_r, axes=2))
    return Fp, Fx


def gaussian_tone(fs: float, T: float, f: float, iota: float, A: float = 1.0, phi0: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time vector t and (h_plus, h_cross) as Gaussian-windowed sin/cos pair."""
    N = int(round(T * fs))
    t = np.arange(N, dtype=float) / fs
    w = np.exp(-0.5 * ((t - T/2.0) / (0.15 * T))**2)
    carrier = 2.0 * math.pi * f * t + phi0
    hp = A * (1.0 + math.cos(iota)**2) * np.cos(carrier) * w
    hx = -2.0 * A * math.cos(iota) * np.sin(carrier) * w
    return t, hp.astype(np.float32), hx.astype(np.float32)


def fractional_delay(sig: np.ndarray, dt: float, fs: float) -> np.ndarray:
    """Apply a fractional time delay via frequency-domain phase ramp."""
    X = rfft(sig)
    freqs = rfftfreq(sig.shape[0], 1.0 / fs)
    X *= np.exp(-1j * 2.0 * np.pi * freqs * dt)
    y = irfft(X, n=sig.shape[0])
    return y.astype(np.float32)


def draw_params(rng: np.random.Generator) -> Tuple[float, float, float, float, float, float]:
    """Draw (theta, phi, psi, iota, f, snr) from broad priors."""
    theta = math.acos(1.0 - 2.0 * rng.random())   # uniform on sphere
    phi   = 2.0 * math.pi * rng.random()
    psi   = 2.0 * math.pi * rng.random()
    iota  = math.acos(2.0 * rng.random() - 1.0)
    f     = 10.0 ** rng.uniform(math.log10(10.0), math.log10(1000.0))
    snr   = rng.uniform(3.0, 100.0)
    return theta, phi, psi, iota, f, snr


def build_event(idx: int, seed: int, fs: float, T: float, rng: np.random.Generator) -> Tuple[EventMeta, np.ndarray, np.ndarray]:
    """Construct one synthetic event; return (meta, strain_H, strain_L)."""
    theta, phi, psi, iota, f, snr = draw_params(rng)
    n_hat = unit_vec_from_sky(theta, phi)
    # antenna patterns
    FH_plus, FH_cross = antenna_pattern(n_hat, psi, u_H, v_H)
    FL_plus, FL_cross = antenna_pattern(n_hat, psi, u_L, v_L)
    # geometric delay (H reference; apply delay at L)
    tau = float(np.dot(n_hat, (r_H - r_L)) / C)
    # waveform
    t, hp, hx = gaussian_tone(fs=fs, T=T, f=f, iota=iota, A=1.0, phi0=rng.uniform(0, 2*math.pi))
    # detector responses
    hH = FH_plus * hp + FH_cross * hx
    hL = FL_plus * fractional_delay(hp, tau, fs) + FL_cross * fractional_delay(hx, tau, fs)
    # noise
    nH = rng.standard_normal(hH.shape).astype(np.float32)
    nL = rng.standard_normal(hL.shape).astype(np.float32)
    # scale to target network SNR (time-domain proxy)
    rhoH = np.linalg.norm(hH) / (np.linalg.norm(nH) + 1e-12)
    rhoL = np.linalg.norm(hL) / (np.linalg.norm(nL) + 1e-12)
    rho0 = math.sqrt(rhoH**2 + rhoL**2) + 1e-12
    scale = snr / rho0
    xH = (scale * hH + nH).astype(np.float32)
    xL = (scale * hL + nL).astype(np.float32)
    meta = EventMeta(
        idx=idx, seed=seed, fs=float(fs), T=float(T), f=float(f), snr=float(snr),
        theta=float(theta), phi=float(phi), psi=float(psi), iota=float(iota),
        tau=float(tau), FH_plus=float(FH_plus), FH_cross=float(FH_cross),
        FL_plus=float(FL_plus), FL_cross=float(FL_cross)
    )
    return meta, xH, xL


def save_event_npz(out_dir: str, meta: EventMeta, xH: np.ndarray, xL: np.ndarray) -> str:
    """Save one event to NPZ and return its path."""
    path = os.path.join(out_dir, f"sample_{meta.idx:06d}.npz")
    np.savez_compressed(
        path,
        strain_H=xH, strain_L=xL,
        fs=meta.fs, T=meta.T, f=meta.f, snr=meta.snr,
        theta=meta.theta, phi=meta.phi, psi=meta.psi, iota=meta.iota,
        tau=meta.tau, FH_plus=meta.FH_plus, FH_cross=meta.FH_cross,
        FL_plus=meta.FL_plus, FL_cross=meta.FL_cross,
        seed=meta.seed, idx=meta.idx
    )
    return path


def write_manifest_csv(out_dir: str, rows: Iterable[EventMeta]) -> None:
    """Write manifest.csv with metadata for all events."""
    path = os.path.join(out_dir, "manifest.csv")
    rows = list(rows)
    if not rows:
        raise ValueError("No rows provided to write_manifest_csv.")
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def sweep_n_and_write(out_dir: str, event_paths: Iterable[str], n_values: Iterable[float], hook_module: str, alpha: float, beta: float) -> None:
    """
    For each event and each n in n_values, call hook.run_localization(event_npz_path, n)
    and collect deltaF_rms, deltatau_rms. Then compute best n* per event by minimizing
    alpha*deltaF_rms + beta*deltatau_rms.
    """
    mod = importlib.import_module(hook_module)
    results_path = os.path.join(out_dir, "results.csv")
    labels_path  = os.path.join(out_dir, "labels.csv")

    # Write per-(event,n) results
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "n", "deltaF_rms", "deltatau_rms"])
        w.writeheader()
        for ev in event_paths:
            for n in n_values:
                out = mod.run_localization(ev, float(n))  # expected keys: deltaF_rms, deltatau_rms
                w.writerow({
                    "event": os.path.basename(ev),
                    "n": float(n),
                    "deltaF_rms": float(out["deltaF_rms"]),
                    "deltatau_rms": float(out["deltatau_rms"]),
                })

    # Compute labels (n*) by reading results
    # Aggregate by event
    by_event = {}
    with open(results_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ev = row["event"]
            n  = float(row["n"])
            dF = float(row["deltaF_rms"])
            dT = float(row["deltatau_rms"])
            score = alpha * dF + beta * dT
            if ev not in by_event or score < by_event[ev][0]:
                by_event[ev] = (score, n, dF, dT)

    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "n_star", "deltaF_rms_at_n_star", "deltatau_rms_at_n_star", "score_alpha_beta"])
        w.writeheader()
        for ev, (score, n_star, dF, dT) in sorted(by_event.items()):
            w.writerow({
                "event": ev,
                "n_star": n_star,
                "deltaF_rms_at_n_star": dF,
                "deltatau_rms_at_n_star": dT,
                "score_alpha_beta": score
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for events and CSVs.")
    ap.add_argument("--num", type=int, default=10_000, help="Number of events to generate.")
    ap.add_argument("--fs", type=float, default=4096.0, help="Sample rate (Hz).")
    ap.add_argument("--T", type=float, default=1.0, help="Duration (s).")
    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed.")
    ap.add_argument("--sweep-n", action="store_true", help="After generation, sweep n by calling a hook module.")
    ap.add_argument("--hook-module", type=str, default="northstar_hook", help="Module containing run_localization(event_path, n).")
    ap.add_argument("--alphabeta", type=float, nargs=2, default=(1.0, 1.0), help="Weights (alpha beta) for combining errors.")
    ap.add_argument("--n-values", type=float, nargs="*", default=[0.5,1,2,4,8,16,32,64], help="Set of n values to test if --sweep-n.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)

    metas = []
    event_paths = []

    for k in range(1, args.num + 1):
        seed_k = args.seed + k
        rng = np.random.default_rng(seed_k)
        meta, xH, xL = build_event(idx=k, seed=seed_k, fs=args.fs, T=args.T, rng=rng)
        path = save_event_npz(args.out, meta, xH, xL)
        metas.append(meta)
        event_paths.append(path)
        if k % 500 == 0:
            print(f"[+] Generated {k}/{args.num} events")

    # write manifest
    write_manifest_csv(args.out, metas)
    print(f"[✓] Wrote manifest.csv with {len(metas)} rows")

    # Optional sweep over n using provided hook
    if args.sweep_n:
        alpha, beta = args.alphabeta
        print(f"[•] Sweeping n values using hook '{args.hook_module}' ...")
        sweep_n_and_write(args.out, event_paths, args.n_values, args.hook_module, alpha, beta)
        print("[✓] Wrote results.csv and labels.csv")

if __name__ == "__main__":
    main()
