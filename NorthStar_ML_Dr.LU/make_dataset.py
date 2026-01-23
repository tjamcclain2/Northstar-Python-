import os
import pandas as pd

OUT = "data/injections"  # <- change to your --out folder

manifest = pd.read_csv(os.path.join(OUT, "manifest.csv"))
labels   = pd.read_csv(os.path.join(OUT, "labels.csv"))

labels["idx"] = labels["event"].str.extract(r"sample_(\d+)\.npz").astype(int)

df = manifest.merge(labels[["idx", "n_star"]], on="idx", how="inner")
df.to_csv(os.path.join(OUT, "dataset_stage1.csv"), index=False)

print("Saved:", os.path.join(OUT, "dataset_stage1.csv"))
print("Rows:", len(df), "Cols:", len(df.columns))
print(df[["idx", "n_star"]].head())
