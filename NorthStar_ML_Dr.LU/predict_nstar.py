import numpy as np
import joblib

BUNDLE = joblib.load("data/injections/rf_nstar.joblib")
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]

def predict_n_star_from_eventmeta(meta_dict):
    # meta_dict should contain keys like manifest.csv columns
    x = np.array([[meta_dict.get(f, 0.0) for f in FEATURES]], dtype=float)
    return float(MODEL.predict(x)[0])
