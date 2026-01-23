# northstar_hook.py
"""
Template hook for simulate_stage1.py --sweep-n.

Implement your connection to the existing localization + weighting code here.
Replace the body of run_localization with a call into your code that accepts:
    - event_npz_path: path to an .npz file produced by the simulator
    - n_value: weighting exponent to use in your algorithm

It must return a dict with keys:
    - "deltaF_rms": float
    - "deltatau_rms": float
"""

from northstar import run_northstar_with_n
import numpy as np

def run_localization(event_npz_path, n_value):
    data = np.load(event_npz_path)
    event = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files}
    
    out = run_northstar_with_n(event, n_value)
    return out

