# Northstar Algorithm Profiling

This repository contains the instrumented version of the **Northstar algorithm**, a Python-based routine for empirical gravitational wave source localization. The code has been prepared for **performance profiling** to identify computational bottlenecks and optimize runtime using Python's built-in profiling tools.

---

## 🔍 What is Profiling?

**Profiling** is the process of analyzing a program to measure:
- Which functions take the most time to run
- How many times each function is called
- Where bottlenecks occur in computation

Profiling helps you **identify slow or inefficient parts** of a program so they can be optimized, parallelized, or offloaded to specialized hardware (e.g., GPUs).

---

## ⚙️ How to Run Profiling with `cProfile` + `SnakeViz`

### ✅ Prerequisites

Install the required packages:
```bash
pip install  snakeviz

```

🚀 Run the Profiler
Use the following command to profile your script and save the output:

```bash
python -m cProfile -o profile_output.prof main.py

```

Then visualize it with:
```bash
snakeviz profile_output.prof
```
