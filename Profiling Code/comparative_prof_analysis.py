import os
import subprocess
import time

def generate_flame_graph(prof_path, output_dir):
    """
    Converts a .prof file into a flame graph PDF using gprof2dot and Graphviz.
    Saves both .dot and .pdf to the output_dir.
    """
    base_name = os.path.splitext(os.path.basename(prof_path))[0]  # e.g., "Original_profile"
    dot_path = os.path.join(output_dir, f"{base_name}.dot")
    pdf_path = os.path.join(output_dir, f"{base_name}_flame.pdf")

    # Step 1: Generate .dot file
    gprof_command = f"gprof2dot -f pstats \"{prof_path}\" -o \"{dot_path}\""
    subprocess.run(gprof_command, shell=True, check=True)

    # Step 2: Convert .dot to .pdf using Graphviz
    graphviz_command = f"dot -Tpdf \"{dot_path}\" -o \"{pdf_path}\""
    subprocess.run(graphviz_command, shell=True, check=True)

    print(f"[✔] Flame graph created: {pdf_path}")

# === Base directory where your .prof files are ===
base_dir = "/Users/jeem/Desktop/Northstar-Python-/Profiling Code"

# === Generate a timestamped subfolder for outputs ===
timestamp = time.strftime("%d_%b_%Y_%H-%M-%S")
output_dir = os.path.join(base_dir, f"flame_graphs_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# === List of .prof files to process ===
profile_files = [
    "Original_profile.prof",
    "Optimized_profile.prof"
]

# === Process each profile ===
for prof_file in profile_files:
    prof_path = os.path.join(base_dir, prof_file)
    generate_flame_graph(prof_path, output_dir)

print(f"\n📂 All flame graphs saved in: {output_dir}")

import pstats
import pandas as pd

def load_stats_to_df(prof_path):
    stats = pstats.Stats(prof_path)
    stats_data = []

    for func, (cc, nc, tt, ct, _) in stats.stats.items():
        filename, line, name = func
        stats_data.append({
            "Function": f"{name} ({filename}:{line})",
            "Calls": cc,
            "Total Time": tt,
            "Cumulative Time": ct
        })

    df = pd.DataFrame(stats_data)
    return df.sort_values("Cumulative Time", ascending=False)

def compare_profiles(original_df, optimized_df, top_n=10):
    merged = pd.merge(
        original_df, optimized_df,
        on="Function", how="outer", suffixes=("_Original", "_Optimized")
    ).fillna(0)

    merged["Cumulative Time Diff"] = (
        merged["Cumulative Time_Original"] - merged["Cumulative Time_Optimized"]
    )
    merged["% Change"] = merged["Cumulative Time Diff"] / merged["Cumulative Time_Original"].replace(0, 1) * 100
    merged = merged.sort_values("Cumulative Time_Original", ascending=False).head(top_n)

    print(f"\n🔍 Top {top_n} Time-Consuming Functions Comparison:\n")
    for _, row in merged.iterrows():
        print(f"▶ {row['Function']}")
        print(f"  - Calls:        {int(row['Calls_Original'])} → {int(row['Calls_Optimized'])}")
        print(f"  - Cum Time:     {row['Cumulative Time_Original']:.4f}s → {row['Cumulative Time_Optimized']:.4f}s")
        print(f"  - Change:       {row['% Change']:.2f}% improvement\n")

# === File paths ===
original_prof = "/Users/jeem/Desktop/Northstar-Python-/Profiling Code/Original_profile.prof"
optimized_prof = "/Users/jeem/Desktop/Northstar-Python-/Profiling Code/Optimized_profile.prof"

# === Load stats ===
original_df = load_stats_to_df(original_prof)
optimized_df = load_stats_to_df(optimized_prof)

# === Compare and print summary ===
compare_profiles(original_df, optimized_df, top_n=10)

